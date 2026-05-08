"""eval_current.py — Batch evaluation using the current streamlit_app.py RAG pipeline.

Replicates the full retrieval logic (routing, decomposition, HyDE, query-rewrite,
LLM reranking) without Streamlit dependencies, loading the same FAISS index and
chunk files as the main app.

Output: results_raw_v4.xlsx  (same schema as results_raw_v3.xlsx)
        question_id | question | config | answer | sources

Usage:
    uv run python external/eval_current.py
    uv run python external/eval_current.py --configs Baseline RAG-Basic RAG-Rerank
    uv run python external/eval_current.py --llm gpt-4.1-mini --out data/result/results_raw_v4.xlsx
"""

import argparse
import json
import os
import re
import sys
import textwrap
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Paths (relative to project root, one level above this file)
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = PROJECT_DIR / "data" / "data_processed"
INDEX_BASE_DIR = PROJECT_DIR / ".rag_index"
QUESTIONS_PATH = PROJECT_DIR / "docs" / "test_questions.json"

CHUNK_FILES = [
    "hk_policy_chunks.json",
    "legco_hansard_chunks.json",
    "Public_Open_Space_chunks.json",
    "Public_Transport_Nodes_chunks.json",
]

# ---------------------------------------------------------------------------
# Evaluation configs
# Each entry may include: use_rag, k, merge_chunks, filter_by_year,
#                         query_rewrite, reranker, pool_factor

CONFIGS: dict[str, dict] = {
    "Baseline": {
        "use_rag": False,
        "k": 10,
        "description": "直接问 LLM，无检索",
    },
    "RAG-Basic": {
        "use_rag": True,
        "k": 10,
        "merge_chunks": True,
        "description": "FAISS k=10，合并相邻 chunk",
    },
    "RAG-Optimized": {
        "use_rag": True,
        "k": 10,
        "merge_chunks": True,
        "filter_by_year": True,
        "description": "RAG-Basic + 按年份降序",
    },
    "RAG-Rewrite": {
        "use_rag": True,
        "k": 10,
        "merge_chunks": True,
        "query_rewrite": True,
        "description": "RAG-Basic + Query Rewrite",
    },
    "RAG-Rerank": {
        "use_rag": True,
        "k": 10,
        "merge_chunks": True,
        "reranker": True,
        "pool_factor": 3,
        "description": "RAG-Basic + LLM Reranker (pool 3×)",
    },
    "RAG-Full": {
        "use_rag": True,
        "k": 10,
        "merge_chunks": True,
        "filter_by_year": True,
        "query_rewrite": True,
        "reranker": True,
        "pool_factor": 3,
        "description": "全优化: Rewrite + Rerank + 年份排序",
    },
    # ---- v2 configs: fixed reranker (larger snippets, no year-sort override) ----
    "RAG-Rerank-v2": {
        "use_rag": True,
        "k": 10,
        "merge_chunks": True,
        "reranker": True,
        "pool_factor": 2,
        "description": "RAG-Basic + 修复后的 LLM Reranker (pool 2×, 500-char snippets)",
    },
    "RAG-Full-v2": {
        "use_rag": True,
        "k": 10,
        "merge_chunks": True,
        "query_rewrite": True,
        "reranker": True,
        "pool_factor": 2,
        "description": "Rewrite + 修复后的 Reranker，去除年份排序干扰",
    },
}

EMBEDDING_CFG = {
    "provider": "openai",
    "model_id": "text-embedding-3-small",
    "index_subdir": "openai_small",
}

# ---------------------------------------------------------------------------
# Document routing constants (mirrors streamlit_app.py)

_PA_YEAR_MAP = {
    "2019": "2019_PolicyAddress.pdf",
    "2020": "2020_PolicyAddress.pdf",
    "2021": "2021_PolicyAddress.pdf",
    "2022": "2022_PolicyAddress.pdf",
    "2023": "2023_PolicyAddress.pdf",
    "2024": "2024_PolicyAddress.pdf",
    "2025": "2025_PolicyAddress.pdf",
}

_BUDGET_FISCAL_MAP = {
    "2019/20": "budget2020.pdf",
    "2020/21": "budget2021.pdf",
    "2021/22": "budget2022.pdf",
    "2022/23": "budget2023.pdf",
    "2023/24": "budget2024.pdf",
    "2024/25": "budget2024.pdf",
    "2025/26": "budget2025.pdf",
    "2026/27": "budget2026.pdf",
}

_PA_KEYWORDS = ["施政報告", "施政报告", "Policy Address"]
_BUDGET_KEYWORDS = ["預算案", "预算案", "财政预算", "財政預算", "Budget"]

# ---------------------------------------------------------------------------
# Cached globals (loaded once per process)
_chunks_data: list | None = None
_faiss_index = None
_metadata: list | None = None
_openai_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


# ---------------------------------------------------------------------------
# Data loading

def load_chunks_data() -> list:
    global _chunks_data
    if _chunks_data is not None:
        return _chunks_data
    all_chunks = []
    for fname in CHUNK_FILES:
        path = DATA_PROCESSED_DIR / fname
        if not path.exists():
            path = PROJECT_DIR / fname
        if not path.exists():
            print(f"  [warn] chunk file not found: {fname}")
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            text = item.get("page_content") or item.get("content", "")
            if not text:
                continue
            meta = item.get("metadata", {}).copy()
            if "page" not in meta:
                meta["page"] = None
            if "year" not in meta:
                date = meta.get("date", "")
                meta["year"] = str(date)[:4] if isinstance(date, str) and date else None
            if "filename" not in meta:
                meta["filename"] = meta.get("source", fname)
            if "source_type" not in meta:
                meta["source_type"] = meta.get("type", "Unknown")
            all_chunks.append({"page_content": text, "metadata": meta})
    _chunks_data = all_chunks
    print(f"  Loaded {len(all_chunks)} chunks from {len(CHUNK_FILES)} files")
    return all_chunks


def load_index() -> tuple:
    global _faiss_index, _metadata
    if _faiss_index is not None:
        return _faiss_index, _metadata
    index_dir = INDEX_BASE_DIR / EMBEDDING_CFG["index_subdir"]
    index_file = index_dir / "index.faiss"
    metadata_file = index_dir / "metadata.json"
    if not index_file.exists():
        sys.exit(f"FAISS index not found at {index_file}. Run: uv run python build_index.py")
    _faiss_index = faiss.read_index(str(index_file))
    with metadata_file.open("r", encoding="utf-8") as f:
        _metadata = json.load(f)
    print(f"  Loaded FAISS index: {_faiss_index.ntotal} vectors from {index_file}")
    return _faiss_index, _metadata


# ---------------------------------------------------------------------------
# Embedding

def embed_query(text: str) -> np.ndarray:
    client = _get_openai_client()
    response = client.embeddings.create(model=EMBEDDING_CFG["model_id"], input=[text])
    vec = np.array([response.data[0].embedding], dtype=np.float32)
    faiss.normalize_L2(vec)
    return vec


# ---------------------------------------------------------------------------
# LLM

def call_llm(prompt: str, llm_model_id: str) -> str:
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=llm_model_id,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


SYSTEM_INSTRUCTIONS = textwrap.dedent("""
    - You are a helpful AI assistant that answers questions about Hong Kong
      government policies, regulations, and public affairs.
    - You will be given extra information provided inside tags like <policy_documents></policy_documents>.
    - CITATION REQUIREMENT: For every specific factual claim (numbers, dates, names,
      statistics, policy details), append an inline citation in the format
      【来源：filename 第X页】. If no matching passage exists in the provided chunks,
      explicitly write (无引用支撑) after that claim.
    - NEGATIVE SPACE RULE: If the retrieved document chunks do NOT contain information
      directly relevant to the question, you MUST state
      "This specific information is not found in the provided documents"
      rather than answering from general knowledge. Do NOT fabricate paragraph numbers,
      page numbers, or statistics.
    - 请用简体中文回答。
""").strip()


def build_answer_prompt(question: str, context: str) -> str:
    parts = [f"<instructions>\n{SYSTEM_INSTRUCTIONS}\n</instructions>"]
    if context:
        parts.append(f"<policy_documents>\n{context}\n</policy_documents>")
    parts.append(f"<question>\n{question}\n</question>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Chunk merging (identical to streamlit_app.py)

def merge_adjacent_chunks(docs: list) -> list:
    if not docs:
        return docs
    grouped: dict = defaultdict(list)
    for doc in docs:
        grouped[doc.get("metadata", {}).get("filename", "unknown")].append(doc)

    merged_docs = []
    for _, doc_list in grouped.items():
        doc_list.sort(
            key=lambda d: d.get("metadata", {}).get("page", 0)
            if d.get("metadata", {}).get("page") is not None else 0
        )
        merged_blocks = []
        current_block = None
        for doc in doc_list:
            page = doc.get("metadata", {}).get("page")
            if page is None:
                merged_blocks.append(doc)
                continue
            try:
                page = int(page)
            except (ValueError, TypeError):
                merged_blocks.append(doc)
                continue
            if current_block is None:
                current_block = {
                    "docs": [doc], "start_page": page, "end_page": page,
                    "texts": [doc.get("page_content", "")],
                }
            else:
                if page - current_block["end_page"] <= 1:
                    current_block["docs"].append(doc)
                    current_block["end_page"] = page
                    current_block["texts"].append(doc.get("page_content", ""))
                else:
                    meta = current_block["docs"][0].get("metadata", {}).copy()
                    meta["page"] = f"{current_block['start_page']}-{current_block['end_page']}"
                    merged_blocks.append({
                        "page_content": "\n\n".join(current_block["texts"]),
                        "metadata": meta,
                    })
                    current_block = {
                        "docs": [doc], "start_page": page, "end_page": page,
                        "texts": [doc.get("page_content", "")],
                    }
        if current_block:
            meta = current_block["docs"][0].get("metadata", {}).copy()
            meta["page"] = f"{current_block['start_page']}-{current_block['end_page']}"
            merged_blocks.append({
                "page_content": "\n\n".join(current_block["texts"]),
                "metadata": meta,
            })
        merged_docs.extend(merged_blocks)
    return merged_docs


# ---------------------------------------------------------------------------
# FAISS search helpers

def _build_retrieved_list(distances, indices_arr, chunks_data, metadata) -> list:
    result = []
    for rank, idx in enumerate(indices_arr[0]):
        if idx < 0:
            continue
        chunk = chunks_data[idx]
        md = metadata[idx] if idx < len(metadata) else {}
        result.append({
            "page_content": chunk.get("page_content", "").strip(),
            "metadata": md,
            "_score": float(distances[0][rank]),
        })
    return result


def faiss_search_full(query: str, k: int) -> list:
    index, metadata = load_index()
    chunks_data = load_chunks_data()
    qv = embed_query(query)
    k = min(k, index.ntotal)
    distances, indices_arr = index.search(qv, k)
    return _build_retrieved_list(distances, indices_arr, chunks_data, metadata)


def faiss_search_filtered(query: str, target_doc: str, k: int) -> list:
    index, metadata = load_index()
    chunks_data = load_chunks_data()
    qv = embed_query(query)

    candidate_indices = [
        i for i, md in enumerate(metadata)
        if md.get("filename", "") == target_doc
    ]
    if not candidate_indices:
        return faiss_search_full(query, k)

    sub_vecs = np.array(
        [index.reconstruct(int(i)) for i in candidate_indices], dtype=np.float32
    )
    sub_index = faiss.IndexFlatIP(sub_vecs.shape[1])
    sub_index.add(sub_vecs)

    sub_k = min(k * 2, len(candidate_indices))
    distances, sub_indices = sub_index.search(qv, sub_k)

    retrieved = []
    for rank, sub_idx in enumerate(sub_indices[0]):
        if sub_idx < 0:
            continue
        orig_idx = candidate_indices[int(sub_idx)]
        chunk = chunks_data[orig_idx]
        md = metadata[orig_idx] if orig_idx < len(metadata) else {}
        retrieved.append({
            "page_content": chunk.get("page_content", "").strip(),
            "metadata": md,
            "_score": float(distances[0][rank]),
        })
    if len(retrieved) < max(2, k // 3):
        return faiss_search_full(query, k)
    return retrieved


# ---------------------------------------------------------------------------
# Document routing

def detect_target_docs(query: str) -> list:
    found = []
    is_pa = any(kw in query for kw in _PA_KEYWORDS)
    is_budget = any(kw in query for kw in _BUDGET_KEYWORDS)

    if is_pa:
        for year in re.findall(r"20\d\d", query):
            fn = _PA_YEAR_MAP.get(year)
            if fn and fn not in found:
                found.append(fn)

    if is_budget:
        fiscal_hits = re.findall(r"20\d\d/\d\d", query)
        for fy in fiscal_hits:
            fn = _BUDGET_FISCAL_MAP.get(fy)
            if fn and fn not in found:
                found.append(fn)
        if not fiscal_hits:
            for year in re.findall(r"20\d\d", query):
                for fy, fn in _BUDGET_FISCAL_MAP.items():
                    if fy.startswith(year) and fn not in found:
                        found.append(fn)

    return found


# ---------------------------------------------------------------------------
# HyDE, Query Rewrite, Decomposition, Reranker

def is_english_query(query: str) -> bool:
    stripped = [c for c in query if not c.isspace()]
    if not stripped:
        return False
    return sum(1 for c in stripped if ord(c) < 128) / len(stripped) > 0.6


def hyde_expand_query(query: str, llm_model_id: str) -> str:
    prompt = (
        "请用1-2句简体中文简要回答以下关于香港政策的问题。"
        "如不确定，可根据常识猜测，重点包含政策相关关键词。\n"
        f"问题：{query}"
    )
    try:
        return call_llm(prompt, llm_model_id)
    except Exception:
        return query


def rewrite_for_retrieval(query: str, llm_model_id: str) -> str:
    prompt = textwrap.dedent(f"""
        你是RAG检索优化助手。请将以下问题改写为向量检索专用的关键词查询。

        改写规则：
        1. 提取核心实体、政策名称、专有名词（保持原文字符）
        2. 保留所有年份、数字、百分比、金额
        3. 去除疑问词（如"是什么"、"有多少"、"如何"）和助词
        4. 可适当补充相关领域关键词，但不要编造具体数字

        原始问题：{query.strip()}

        改写后的检索查询（一句关键词，不要解释）：
    """)
    try:
        result = call_llm(prompt, llm_model_id).strip()
        if result and len(result) < len(query) * 3:
            return result
    except Exception:
        pass
    return query


def decompose_cross_doc_query(query: str, target_docs: list, llm_model_id: str) -> list:
    doc_list = "\n".join(f"- {d}" for d in target_docs)
    prompt = textwrap.dedent(f"""
        你是RAG检索助手。请将以下跨文档问题分解为针对各文档的子查询，以便分别检索最相关内容。

        原始问题：{query}

        相关文档：
        {doc_list}

        请输出JSON数组（仅输出JSON，不要解释）：
        [
          {{"target_doc": "文件名.pdf", "subquery": "针对该文档的检索关键词和问题"}},
          ...
        ]
        每个子查询应聚焦于原问题在该文档中的相关部分，使用利于向量检索的关键词。
    """)
    try:
        response = call_llm(prompt, llm_model_id)
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            if isinstance(parsed, list) and all("target_doc" in x and "subquery" in x for x in parsed):
                return parsed
    except Exception:
        pass
    return [{"target_doc": d, "subquery": query} for d in target_docs]


def llm_rerank(query: str, chunks: list, top_k: int, llm_model_id: str) -> list:
    if not chunks or len(chunks) <= top_k:
        return chunks[:top_k]
    # Larger snippet budget: 500 chars minimum, up to 10000/n per chunk
    snippet_len = max(500, 10000 // len(chunks))
    lines = [
        f"[{i + 1}] [file={c.get('metadata', {}).get('filename', '?')} "
        f"year={c.get('metadata', {}).get('year', '?')} "
        f"page={c.get('metadata', {}).get('page', '?')}]\n{c['page_content'][:snippet_len]}"
        for i, c in enumerate(chunks)
    ]
    prompt = textwrap.dedent(f"""
        你是信息检索评估助手。请评估以下每个文档片段与问题的相关性。

        评分标准（0-10整数）：
        - 10分：片段直接包含问题所需的具体数字、时间节点或政策名称
        - 7-9分：片段包含高度相关的背景信息，可辅助回答问题
        - 4-6分：片段与问题主题相关，但缺少关键细节
        - 1-3分：片段仅主题沾边，内容对回答问题帮助不大
        - 0分：片段与问题完全无关

        注意：优先给来自问题中明确提及的文件和年份的片段更高分。

        问题：{query}

        文档片段：
        {"\\ \n\n".join(lines)}

        请输出JSON数组，每个元素包含 idx（1起始）和 score（0-10整数）：
        [{{"idx": 1, "score": 8}}, {{"idx": 2, "score": 3}}, ...]
        仅输出JSON，不要任何解释。
    """)
    try:
        response = call_llm(prompt, llm_model_id)
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            scores_data = json.loads(match.group())
            score_map = {
                int(s["idx"]) - 1: float(s["score"])
                for s in scores_data
                if isinstance(s, dict) and "idx" in s and "score" in s
            }
            scored = sorted(enumerate(chunks), key=lambda x: score_map.get(x[0], 0.0), reverse=True)
            return [c for _, c in scored[:top_k]]
    except Exception:
        pass
    return chunks[:top_k]


# ---------------------------------------------------------------------------
# Main retrieval function

def retrieve(query: str, config: dict, llm_model_id: str) -> tuple[list, str]:
    """Returns (retrieved_chunks, search_query_used)."""
    k = config.get("k", 10)
    pool_factor = config.get("pool_factor", 1)
    fetch_k = k * pool_factor if config.get("reranker") else k

    search_query = query

    # Query Rewrite
    if config.get("query_rewrite"):
        search_query = rewrite_for_retrieval(query, llm_model_id)

    # HyDE for English
    if is_english_query(search_query):
        search_query = hyde_expand_query(search_query, llm_model_id)

    # Document routing
    target_docs = detect_target_docs(query)

    if len(target_docs) >= 2:
        sub_queries = decompose_cross_doc_query(query, target_docs, llm_model_id)
        sub_k = max(3, fetch_k // max(len(sub_queries), 1))
        seen: set = set()
        retrieved = []
        for sq in sub_queries:
            for r in faiss_search_filtered(sq["subquery"], sq["target_doc"], sub_k):
                key = hash(r["page_content"][:120])
                if key not in seen:
                    seen.add(key)
                    retrieved.append(r)
        # Top-up if needed
        if len(retrieved) < fetch_k:
            for r in faiss_search_full(search_query, fetch_k):
                key = hash(r["page_content"][:120])
                if key not in seen:
                    seen.add(key)
                    retrieved.append(r)
                    if len(retrieved) >= fetch_k:
                        break

    elif len(target_docs) == 1:
        retrieved = faiss_search_filtered(search_query, target_docs[0], fetch_k)

    else:
        retrieved = faiss_search_full(search_query, fetch_k)

    # Reranking
    if config.get("reranker") and len(retrieved) > k:
        retrieved = llm_rerank(query, retrieved, k, llm_model_id)

    # Post-processing
    if config.get("filter_by_year"):
        retrieved = sorted(
            retrieved,
            key=lambda d: d["metadata"].get("year", "0") or "0",
            reverse=True,
        )

    if config.get("merge_chunks"):
        retrieved = merge_adjacent_chunks(retrieved)

    return retrieved, search_query


def format_context(chunks: list) -> str:
    lines = []
    for i, doc in enumerate(chunks):
        md = doc.get("metadata", {})
        lines.append(
            "\n".join([
                f"[Chunk {i + 1}]",
                f"source_type={md.get('source_type', '?')}; year={md.get('year', '?')}; "
                f"file={md.get('filename', '?')}; page={md.get('page', '?')}",
                doc.get("page_content", "").strip(),
            ])
        )
    return "\n\n".join(lines)


def format_sources(chunks: list) -> str:
    parts = []
    for doc in chunks:
        md = doc.get("metadata", {})
        fn = md.get("filename", "?")
        pg = md.get("page", "?")
        parts.append(f"{fn} 第{pg}页")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Per-question answering

def answer_question(question: str, config: dict, llm_model_id: str) -> tuple[str, str]:
    """Returns (answer_text, sources_str)."""
    if not config.get("use_rag"):
        prompt = build_answer_prompt(question, "")
        return call_llm(prompt, llm_model_id), ""

    chunks, _ = retrieve(question, config, llm_model_id)
    context = format_context(chunks)
    sources = format_sources(chunks)
    prompt = build_answer_prompt(question, context)
    answer = call_llm(prompt, llm_model_id)
    return answer, sources


# ---------------------------------------------------------------------------
# Main

def parse_args():
    parser = argparse.ArgumentParser(description="Batch evaluation with current RAG pipeline")
    parser.add_argument(
        "--configs", nargs="+", default=list(CONFIGS.keys()),
        choices=list(CONFIGS.keys()),
        help="Configs to run (default: all)",
    )
    parser.add_argument(
        "--llm", default="gpt-4o-mini",
        help="LLM model id (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--out", default=str(PROJECT_DIR / "data" / "result" / "results_raw_v5.xlsx"),
        help="Output Excel path",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Seconds between API calls (default: 0.5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"=== eval_current.py ===")
    print(f"  LLM:     {args.llm}")
    print(f"  Configs: {args.configs}")
    print(f"  Output:  {args.out}")

    # Pre-load shared resources
    print("\nLoading resources...")
    load_chunks_data()
    load_index()

    # Load questions
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"  Questions: {len(questions)}")

    results = []
    total = len(questions) * len(args.configs)
    done = 0

    for q_item in questions:
        q_id = q_item.get("id", "?")
        question_text = q_item.get("question", "")
        if not question_text:
            continue

        for config_name in args.configs:
            config = CONFIGS[config_name]
            done += 1
            print(f"[{done}/{total}] {q_id} | {config_name} | {question_text[:50]}...")

            try:
                answer, sources = answer_question(question_text, config, args.llm)
                results.append({
                    "question_id": q_id,
                    "question": question_text,
                    "config": config_name,
                    "answer": answer,
                    "sources": sources,
                })
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "question_id": q_id,
                    "question": question_text,
                    "config": config_name,
                    "answer": f"ERROR: {e}",
                    "sources": "",
                })

            time.sleep(args.delay)

    df = pd.DataFrame(results)
    df = df.sort_values(["question_id", "config"]).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False)
    print(f"\n✅ Done. Results saved to {out_path} ({len(df)} rows)")

    # Print quick summary
    print("\n--- Row count by config ---")
    for cfg, cnt in df.groupby("config").size().items():
        print(f"  {cfg}: {cnt} rows")


if __name__ == "__main__":
    main()

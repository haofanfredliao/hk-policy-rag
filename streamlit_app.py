from htbuilder.units import rem
from htbuilder import div, styles
from collections import namedtuple, defaultdict
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
import os
from pathlib import Path
import textwrap
import time

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

st.set_page_config(page_title="HK Policy AI Assistant", page_icon="✨", layout="wide")

# -----------------------------------------------------------------------------
# Constants & Paths

executor = ThreadPoolExecutor(max_workers=5)

HISTORY_LENGTH = 5
SUMMARIZE_OLD_HISTORY = True
DOCS_CONTEXT_LEN = 10
EXTRA_CONTEXT_LEN = 10
MIN_TIME_BETWEEN_REQUESTS = datetime.timedelta(seconds=3)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PROCESSED_DIR = PROJECT_DIR / "data" / "data_processed"
INDEX_BASE_DIR = PROJECT_DIR / ".rag_index"
EMBEDDING_BATCH_SIZE = 100

DEBUG_MODE = st.query_params.get("debug", "false").lower() == "true"

CHUNK_FILES = [
    "hk_policy_chunks.json",
    "legco_hansard_chunks.json",
    "Public_Open_Space_chunks.json",
    "Public_Transport_Nodes_chunks.json",
]

# -----------------------------------------------------------------------------
# Model & Config Registries  (language-neutral internal keys)

LLM_MODELS = {
    "gpt-4o-mini":  {"provider": "openai", "model_id": "gpt-4o-mini"},
    "gpt-4o":       {"provider": "openai", "model_id": "gpt-4o"},
    "gpt-4.1":      {"provider": "openai", "model_id": "gpt-4.1"},
    "gpt-4.1-mini": {"provider": "openai", "model_id": "gpt-4.1-mini"},
    "grok-3":       {"provider": "xai",    "model_id": "grok-3"},
    "grok-3-fast":  {"provider": "xai",    "model_id": "grok-3-fast"},
}

LLM_MODEL_LABELS = {
    "en": {
        "gpt-4o-mini":  "GPT-4o mini (Default)",
        "gpt-4o":       "GPT-4o",
        "gpt-4.1":      "GPT-4.1",
        "gpt-4.1-mini": "GPT-4.1 mini",
        "grok-3":       "xAI Grok 3",
        "grok-3-fast":  "xAI Grok 3 Fast",
    },
    "zh-cn": {
        "gpt-4o-mini":  "GPT-4o mini（默认）",
        "gpt-4o":       "GPT-4o",
        "gpt-4.1":      "GPT-4.1",
        "gpt-4.1-mini": "GPT-4.1 mini",
        "grok-3":       "xAI Grok 3",
        "grok-3-fast":  "xAI Grok 3 Fast",
    },
    "zh-tw": {
        "gpt-4o-mini":  "GPT-4o mini（預設）",
        "gpt-4o":       "GPT-4o",
        "gpt-4.1":      "GPT-4.1",
        "gpt-4.1-mini": "GPT-4.1 mini",
        "grok-3":       "xAI Grok 3",
        "grok-3-fast":  "xAI Grok 3 Fast",
    },
}

EMBEDDING_MODELS = {
    "openai-small": {
        "provider": "openai",
        "model_id": "text-embedding-3-small",
        "dim": 1536,
        "index_subdir": "openai_small",
    },
    "openai-large": {
        "provider": "openai",
        "model_id": "text-embedding-3-large",
        "dim": 3072,
        "index_subdir": "openai_large",
    },
    "bge-small-zh": {
        "provider": "huggingface",
        "model_id": "BAAI/bge-small-zh-v1.5",
        "dim": 512,
        "index_subdir": "hf_bge_small_zh",
    },
    "bge-m3": {
        "provider": "huggingface",
        "model_id": "BAAI/bge-m3",
        "dim": 1024,
        "index_subdir": "hf_bge_m3",
    },
}

EMBEDDING_MODEL_LABELS = {
    "en": {
        "openai-small": "OpenAI text-embedding-3-small (Default)",
        "openai-large": "OpenAI text-embedding-3-large",
        "bge-small-zh": "BAAI/bge-small-zh-v1.5 (Open Source)",
        "bge-m3":       "BAAI/bge-m3 (Open Source, Multilingual)",
    },
    "zh-cn": {
        "openai-small": "OpenAI text-embedding-3-small（默认）",
        "openai-large": "OpenAI text-embedding-3-large",
        "bge-small-zh": "BAAI/bge-small-zh-v1.5（开源）",
        "bge-m3":       "BAAI/bge-m3（开源多语言）",
    },
    "zh-tw": {
        "openai-small": "OpenAI text-embedding-3-small（預設）",
        "openai-large": "OpenAI text-embedding-3-large",
        "bge-small-zh": "BAAI/bge-small-zh-v1.5（開源）",
        "bge-m3":       "BAAI/bge-m3（開源多語言）",
    },
}

RAG_CONFIGS = {
    "rag-optimized": {
        "use_rag": True,
        "k": 10,
        "merge_chunks": True,
        "filter_by_year": True,
    },
    "rag-basic": {
        "use_rag": True,
        "k": 10,
        "merge_chunks": True,
        "filter_by_year": False,
    },
    "baseline": {
        "use_rag": False,
        "k": 0,
        "merge_chunks": False,
        "filter_by_year": False,
    },
}

RAG_CONFIG_LABELS = {
    "en": {
        "rag-optimized": "RAG-Optimized (Recommended)",
        "rag-basic":     "RAG-Basic",
        "baseline":      "Baseline (No Retrieval)",
    },
    "zh-cn": {
        "rag-optimized": "RAG-优化版（推荐）",
        "rag-basic":     "RAG-基础版",
        "baseline":      "Baseline（无检索）",
    },
    "zh-tw": {
        "rag-optimized": "RAG-優化版（推薦）",
        "rag-basic":     "RAG-基礎版",
        "baseline":      "Baseline（無檢索）",
    },
}

RAG_CONFIG_DESCRIPTIONS = {
    "en": {
        "rag-optimized": "RAG optimised: k=10, merge adjacent chunks, sort by year",
        "rag-basic":     "RAG basic: k=10, merge adjacent chunks",
        "baseline":      "Direct LLM query — no document retrieval",
    },
    "zh-cn": {
        "rag-optimized": "RAG 优化配置：k=10，合并相邻 chunk，按年份排序",
        "rag-basic":     "RAG 基础配置：k=10，合并相邻 chunk",
        "baseline":      "直接问 LLM，无文档检索",
    },
    "zh-tw": {
        "rag-optimized": "RAG 優化配置：k=10，合併相鄰 chunk，按年份排序",
        "rag-basic":     "RAG 基礎配置：k=10，合併相鄰 chunk",
        "baseline":      "直接問 LLM，無文件檢索",
    },
}

# -----------------------------------------------------------------------------
# i18n: UI String Translations

TRANSLATIONS = {
    "en": {
        # Sidebar
        "lang_label":           "🌐 Language",
        "sidebar_header":       "⚙️ Settings",
        "llm_section":          "🤖 AI Model",
        "xai_warning":          "Set GROK_API_KEY in .env to use xAI models.",
        "rag_section":          "🔍 RAG Retrieval Mode",
        "rag_k_label":          "Retrieved docs (k)",
        "rag_merge_label":      "Merge adjacent chunks",
        "rag_sort_label":       "Sort by year (newest first)",
        "embedding_section":    "📐 Embedding Model",
        "hf_info":              "Open-source model will be downloaded from HuggingFace on first use. This may take a few minutes.",
        "index_ready":          "Index ready",
        "index_missing":        "Index not built. Will be built automatically on first query.",
        "metrics_section":      "📊 Evaluation Metrics",
        "metrics_caption":      "Based on 60 test questions (Grok + HuggingFace Embedding):",
        "metrics_note":         "Baseline accuracy only 3%. RAG significantly improves answer quality.",
        # Main UI
        "page_title":           "HK Policy AI Assistant",
        "chat_placeholder":     "Ask a question...",
        "chat_followup":        "Ask a follow-up...",
        "restart_button":       "Restart",
        "disclaimer_btn":       "&nbsp;:small[:gray[:material/balance: Legal disclaimer]]",
        # Spinners
        "spinner_waiting":      "Waiting...",
        "spinner_researching":  "Researching... ({rag})",
        "spinner_thinking":     "Thinking... ({llm})",
        "spinner_prompt":       "Computing prompt...",
        "spinner_prompt_done":  "Prompt computed",
        # Feedback
        "feedback_btn":         "How did I do?",
        "feedback_rating":      "Rating",
        "feedback_detail":      "More information (optional)",
        "feedback_history":     "Include chat history with my feedback",
        "feedback_submit":      "Send feedback",
        # Chunks expander
        "chunks_expander":      "Referenced knowledge chunks",
        # Disclaimer dialog
        "disclaimer_title":     "Legal disclaimer",
        "disclaimer_text": (
            "This AI chatbot is powered by OpenAI and local policy documents. "
            "Answers may be inaccurate, inefficient, or biased. "
            "Any use or decisions based on such answers should include reasonable "
            "practices including human oversight to ensure they are safe, accurate, "
            "and suitable for your intended purpose. The project is not liable for "
            "any actions, losses, or damages resulting from the use of the chatbot. "
            "Do not enter any private, sensitive, personal, or regulated data."
        ),
        # Retrieval Optimisation section
        "retrieval_opt_section":    "⚡ Retrieval Optimisation",
        "query_rewrite_label":      "Query rewrite (improves keyword match)",
        "reranker_label":           "LLM reranker (scores answer-relevance)",
        "pool_factor_label":        "Candidate pool multiplier",
        "pool_factor_help":         "Retrieve k × N candidates before reranking, then keep top k.",
        # LLM instruction language directive
        "lang_directive": "",
    },
    "zh-cn": {
        "lang_label":           "🌐 语言",
        "sidebar_header":       "⚙️ 模型配置",
        "llm_section":          "🤖 AI 基座模型",
        "xai_warning":          "请在 .env 中设置 GROK_API_KEY 以使用 xAI 模型。",
        "rag_section":          "🔍 RAG 检索模式",
        "rag_k_label":          "检索文档数 k",
        "rag_merge_label":      "合并相邻 chunk",
        "rag_sort_label":       "按年份排序（优先最新）",
        "embedding_section":    "📐 Embedding 模型",
        "hf_info":              "开源模型首次使用时需从 HuggingFace 下载，可能需要几分钟。",
        "index_ready":          "索引已就绪",
        "index_missing":        "索引未构建，首次提问时将自动构建。",
        "metrics_section":      "📊 评测参考指标",
        "metrics_caption":      "基于 60 道测试题的评测结果（Grok + HuggingFace Embedding）：",
        "metrics_note":         "Baseline 准确率仅 3%，RAG 显著提升回答质量。",
        "page_title":           "香港政策 AI 助手",
        "chat_placeholder":     "请输入问题…",
        "chat_followup":        "继续提问…",
        "restart_button":       "重新开始",
        "disclaimer_btn":       "&nbsp;:small[:gray[:material/balance: 法律免责声明]]",
        "spinner_waiting":      "请稍候…",
        "spinner_researching":  "正在检索… ({rag})",
        "spinner_thinking":     "正在思考… ({llm})",
        "spinner_prompt":       "正在构建提示词…",
        "spinner_prompt_done":  "提示词已构建",
        "feedback_btn":         "回答如何？",
        "feedback_rating":      "评分",
        "feedback_detail":      "更多信息（可选）",
        "feedback_history":     "附上对话历史",
        "feedback_submit":      "提交反馈",
        "chunks_expander":      "参考知识块",
        "disclaimer_title":     "法律免责声明",
        "disclaimer_text": (
            "本 AI 聊天机器人由 OpenAI 和本地政策文件驱动。"
            "回答可能不准确、效率低下或存在偏差。"
            "任何基于此回答的使用或决策均应采取合理措施，包括人工监督，"
            "以确保其安全、准确并适合预期用途。"
            "本项目不对因使用本聊天机器人而导致的任何行为、损失或损害承担责任。"
            "请勿输入任何私人、敏感、个人或受监管的数据。"
        ),
        # Retrieval Optimisation section
        "retrieval_opt_section":    "⚡ 检索优化",
        "query_rewrite_label":      "查询改写（提升关键词匹配）",
        "reranker_label":           "LLM 精排（评估答案相关性）",
        "pool_factor_label":        "候选扩大倍数",
        "pool_factor_help":         "先取 k × N 个候选，精排后保留前 k 个。",
        "lang_directive": "请用简体中文回答。",
    },
    "zh-tw": {
        "lang_label":           "🌐 語言",
        "sidebar_header":       "⚙️ 模型配置",
        "llm_section":          "🤖 AI 基礎模型",
        "xai_warning":          "請在 .env 中設定 GROK_API_KEY 以使用 xAI 模型。",
        "rag_section":          "🔍 RAG 檢索模式",
        "rag_k_label":          "檢索文件數 k",
        "rag_merge_label":      "合併相鄰 chunk",
        "rag_sort_label":       "按年份排序（優先最新）",
        "embedding_section":    "📐 Embedding 模型",
        "hf_info":              "開源模型首次使用時需從 HuggingFace 下載，可能需要幾分鐘。",
        "index_ready":          "索引已就緒",
        "index_missing":        "索引未構建，首次提問時將自動構建。",
        "metrics_section":      "📊 評測參考指標",
        "metrics_caption":      "基於 60 道測試題的評測結果（Grok + HuggingFace Embedding）：",
        "metrics_note":         "Baseline 準確率僅 3%，RAG 顯著提升回答質量。",
        "page_title":           "香港政策 AI 助手",
        "chat_placeholder":     "請輸入問題…",
        "chat_followup":        "繼續提問…",
        "restart_button":       "重新開始",
        "disclaimer_btn":       "&nbsp;:small[:gray[:material/balance: 法律免責聲明]]",
        "spinner_waiting":      "請稍候…",
        "spinner_researching":  "正在檢索… ({rag})",
        "spinner_thinking":     "正在思考… ({llm})",
        "spinner_prompt":       "正在構建提示詞…",
        "spinner_prompt_done":  "提示詞已構建",
        "feedback_btn":         "回答如何？",
        "feedback_rating":      "評分",
        "feedback_detail":      "更多資訊（可選）",
        "feedback_history":     "附上對話歷史",
        "feedback_submit":      "提交反饋",
        "chunks_expander":      "參考知識塊",
        "disclaimer_title":     "法律免責聲明",
        "disclaimer_text": (
            "本 AI 聊天機器人由 OpenAI 及本地政策文件驅動。"
            "回答可能不準確、效率低下或存在偏差。"
            "任何基於此回答的使用或決策均應採取合理措施，包括人工監督，"
            "以確保其安全、準確並適合預期用途。"
            "本項目不對因使用本聊天機器人而導致的任何行為、損失或損害承擔責任。"
            "請勿輸入任何私人、敏感、個人或受監管的資料。"
        ),
        # Retrieval Optimisation section
        "retrieval_opt_section":    "⚡ 檢索優化",
        "query_rewrite_label":      "查詢改寫（提升關鍵詞匹配）",
        "reranker_label":           "LLM 精排（評估答案相關性）",
        "pool_factor_label":        "候選擴大倍數",
        "pool_factor_help":         "先取 k × N 個候選，精排後保留前 k 個。",
        "lang_directive": "請以繁體中文回答。",
    },
}

# -----------------------------------------------------------------------------
# Suggested questions per language (chosen from eval results with total_score >= 5)

SUGGESTIONS_BY_LANG = {
    "en": {
        ":blue[:material/apartment:] Lok Ma Chau Loop": (
            "According to the 2024 Policy Address, to what total floor area was Phase 1 of the Lok Ma Chau Loop (Hong Kong Park) expanded?"
        ),
        ":green[:material/home:] Early Move-in Scheme": (
            "According to the 2024 Policy Address, how many families have benefited from the Early Move-in Scheme, and how much rental expenditure has been saved for beneficiaries?"
        ),
        ":orange[:material/work:] GBA Youth Employment": (
            "According to the 2024 Policy Address, what adjustments were made to the Greater Bay Area Youth Employment Scheme, and what is the new employment allowance cap?"
        ),
        ":violet[:material/science:] New Industrialisation": (
            "According to the 2026-27 Budget, how many new smart production lines has the New Industrialisation Funding Scheme supported, and how much private investment has been leveraged?"
        ),
        ":red[:material/account_balance:] Head of Dept. Accountability": (
            "According to the 2025 Policy Address, what is the purpose of the Head of Department Accountability System? How is the investigation mechanism triggered, and what is the Independent Review Panel?"
        ),
    },
    "zh-cn": {
        ":blue[:material/apartment:] 河套园区进展": (
            "2024年施政报告中，河套香港园区第一期总楼面面积倍增至多少平方米？"
        ),
        ":green[:material/home:] 提前上楼计划": (
            "2024年施政报告提到，「提前上楼计划」至报告发布时已为多少个家庭提前上楼？为受惠者节省了约多少租金开支？"
        ),
        ":orange[:material/work:] 大湾区青年就业": (
            "根据2024年施政报告，「大湾区青年就业计划」有何新调整？就业津贴上限调整为多少元？"
        ),
        ":violet[:material/science:] 新型工业化": (
            "2026/27年度财政预算案中，「新型工业化资助计划」支持了多少条新智能生产线？带动多少私人投资？"
        ),
        ":red[:material/account_balance:] 部门首长责任制": (
            "2025年施政报告提出「部门首长责任制」，目的是什么？调查机制如何启动？独立调查小组是什么？"
        ),
    },
    "zh-tw": {
        ":blue[:material/apartment:] 河套園區進展": (
            "2024年施政報告中，河套香港園區第一期總樓面面積倍增至多少平方米？"
        ),
        ":green[:material/home:] 提前上樓計劃": (
            "2024年施政報告提到，「提前上樓計劃」至報告發布時已為多少個家庭提前上樓？為受惠者節省了約多少租金開支？"
        ),
        ":orange[:material/work:] 大灣區青年就業": (
            "根據2024年施政報告，「大灣區青年就業計劃」有何新調整？就業津貼上限調整為多少元？"
        ),
        ":violet[:material/science:] 新型工業化": (
            "2026/27年度財政預算案中，「新型工業化資助計劃」支持了多少條新智能生產線？帶動多少私人投資？"
        ),
        ":red[:material/account_balance:] 部門首長責任制": (
            "2025年施政報告提出「部門首長責任制」，目的是什麼？調查機制如何啟動？獨立調查小組是什麼？"
        ),
    },
}

# -----------------------------------------------------------------------------
# Language selector (must render before the rest of the sidebar)

with st.sidebar:
    selected_lang = st.selectbox(
        "🌐 Language",
        options=["en", "zh-cn", "zh-tw"],
        format_func=lambda x: {"en": "English", "zh-cn": "简体中文", "zh-tw": "繁體中文"}[x],
        index=0,
        key="lang",
        label_visibility="collapsed",
    )

t = TRANSLATIONS[selected_lang]
suggestions = SUGGESTIONS_BY_LANG[selected_lang]

# -----------------------------------------------------------------------------
# RAG Logic (ported from batch_eval.py)

def merge_adjacent_chunks(docs):
    """Merges adjacent chunks from the same document."""
    if not docs:
        return docs

    grouped = defaultdict(list)
    for doc in docs:
        filename = doc.get("metadata", {}).get("filename", "unknown")
        grouped[filename].append(doc)

    merged_docs = []
    for filename, doc_list in grouped.items():
        doc_list.sort(
            key=lambda d: d.get("metadata", {}).get("page", 0)
            if d.get("metadata", {}).get("page") is not None
            else 0
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
                    "docs": [doc],
                    "start_page": page,
                    "end_page": page,
                    "texts": [doc.get("page_content", "")],
                }
            else:
                if page - current_block["end_page"] <= 1:
                    current_block["docs"].append(doc)
                    current_block["end_page"] = page
                    current_block["texts"].append(doc.get("page_content", ""))
                else:
                    merged_text = "\n\n".join(current_block["texts"])
                    merged_meta = current_block["docs"][0].get("metadata", {}).copy()
                    merged_meta["start_page"] = current_block["start_page"]
                    merged_meta["end_page"] = current_block["end_page"]
                    merged_meta["page"] = f"{current_block['start_page']}-{current_block['end_page']}"
                    merged_blocks.append({"page_content": merged_text, "metadata": merged_meta})
                    current_block = {
                        "docs": [doc],
                        "start_page": page,
                        "end_page": page,
                        "texts": [doc.get("page_content", "")],
                    }

        if current_block:
            merged_text = "\n\n".join(current_block["texts"])
            merged_meta = current_block["docs"][0].get("metadata", {}).copy()
            merged_meta["start_page"] = current_block["start_page"]
            merged_meta["end_page"] = current_block["end_page"]
            merged_meta["page"] = f"{current_block['start_page']}-{current_block['end_page']}"
            merged_blocks.append({"page_content": merged_text, "metadata": merged_meta})

        merged_docs.extend(merged_blocks)

    return merged_docs


# -----------------------------------------------------------------------------
# Embedding Functions

def _embed_texts_openai(texts, model_id="text-embedding-3-small"):
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = openai_client.embeddings.create(model=model_id, input=texts)
    return [item.embedding for item in response.data]


def _embed_texts_huggingface(texts, model_id="BAAI/bge-small-zh-v1.5"):
    from langchain_huggingface import HuggingFaceEmbeddings
    hf_model = HuggingFaceEmbeddings(model_name=model_id)
    return hf_model.embed_documents(texts)


def embed_texts(texts, embedding_cfg):
    if embedding_cfg["provider"] == "openai":
        return _embed_texts_openai(texts, embedding_cfg["model_id"])
    elif embedding_cfg["provider"] == "huggingface":
        return _embed_texts_huggingface(texts, embedding_cfg["model_id"])
    raise ValueError(f"Unknown embedding provider: {embedding_cfg['provider']}")


def embed_query(query, embedding_cfg):
    if embedding_cfg["provider"] == "openai":
        return _embed_texts_openai([query], embedding_cfg["model_id"])[0]
    elif embedding_cfg["provider"] == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        hf_model = HuggingFaceEmbeddings(model_name=embedding_cfg["model_id"])
        return hf_model.embed_query(query)
    raise ValueError(f"Unknown embedding provider: {embedding_cfg['provider']}")


# -----------------------------------------------------------------------------
# Data Loading

@st.cache_data
def get_all_chunks_data():
    """Loads all chunk JSON files from data/data_processed/."""
    all_chunks = []
    for fname in CHUNK_FILES:
        path = DATA_PROCESSED_DIR / fname
        if not path.exists():
            path = PROJECT_DIR / fname
        if not path.exists():
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
                if "date" in meta and isinstance(meta["date"], str):
                    meta["year"] = meta["date"][:4]
                else:
                    meta["year"] = None
            if "filename" not in meta:
                meta["filename"] = meta.get("source", fname)
            if "source_type" not in meta:
                meta["source_type"] = meta.get("type", "Unknown")
            all_chunks.append({"page_content": text, "metadata": meta})
    return all_chunks


def _get_index_paths(embedding_cfg):
    index_dir = INDEX_BASE_DIR / embedding_cfg["index_subdir"]
    return index_dir, index_dir / "index.faiss", index_dir / "metadata.json"


def _build_and_persist_index(chunks_data, embedding_cfg):
    if not chunks_data:
        return None, []

    index_dir, index_file, metadata_file = _get_index_paths(embedding_cfg)
    index_dir.mkdir(parents=True, exist_ok=True)

    texts = [item.get("page_content", "") for item in chunks_data]
    metadata = [item.get("metadata", {}) for item in chunks_data]

    all_embeddings = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i: i + EMBEDDING_BATCH_SIZE]
        all_embeddings.extend(embed_texts(batch, embedding_cfg))

    vectors = np.array(all_embeddings, dtype=np.float32)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(index_file))
    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    return index, metadata


@st.cache_resource
def get_rag_index(embedding_model_key: str):
    """Loads or builds a FAISS index for the given embedding model."""
    embedding_cfg = EMBEDDING_MODELS[embedding_model_key]
    if embedding_cfg["provider"] == "openai" and not os.getenv("OPENAI_API_KEY"):
        return None, []

    _, index_file, metadata_file = _get_index_paths(embedding_cfg)
    if index_file.exists() and metadata_file.exists():
        index = faiss.read_index(str(index_file))
        with metadata_file.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        return index, metadata

    chunks_data = get_all_chunks_data()
    return _build_and_persist_index(chunks_data, embedding_cfg)


# -----------------------------------------------------------------------------
# LLM Helpers

def call_llm_stream(prompt, llm_cfg):
    """Calls the selected LLM and returns a streaming text generator."""
    provider = llm_cfg["provider"]
    model_id = llm_cfg["model_id"]

    if provider == "openai":
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        stream = openai_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        def _gen():
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        return _gen()

    elif provider == "xai":
        from xai_sdk import Client as XaiClient
        from xai_sdk.chat import system, user as xai_user
        api_key = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
        xai_client = XaiClient(api_key=api_key)
        chat = xai_client.chat.create(model=model_id)
        chat.append(system("You are a professional and accurate Hong Kong policy assistant."))
        chat.append(xai_user(prompt))
        response = chat.sample()
        text = response.content

        def _gen():
            yield text

        return _gen()

    raise ValueError(f"Unknown provider: {provider}")


def call_llm_once(prompt, llm_cfg):
    """Calls the selected LLM (non-streaming) for summarisation tasks."""
    provider = llm_cfg["provider"]
    model_id = llm_cfg["model_id"]

    if provider == "openai":
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = openai_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    elif provider == "xai":
        from xai_sdk import Client as XaiClient
        from xai_sdk.chat import system, user as xai_user
        api_key = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
        xai_client = XaiClient(api_key=api_key)
        chat = xai_client.chat.create(model=model_id)
        chat.append(system("You are a professional and accurate Hong Kong policy assistant."))
        chat.append(xai_user(prompt))
        return chat.sample().content

    raise ValueError(f"Unknown provider: {provider}")


# -----------------------------------------------------------------------------
# Prompt Building

def build_prompt(**kwargs):
    prompt = []
    for name, contents in kwargs.items():
        if contents:
            prompt.append(f"<{name}>\n{contents}\n</{name}>")
    return "\n".join(prompt)


def get_instructions(lang):
    base = textwrap.dedent("""
        - You are a helpful AI assistant that answers questions about Hong Kong
          government policies, regulations, and public affairs.
        - You will be given extra information provided inside tags like this
          <foo></foo>.
        - Use context and history to provide a coherent answer.
        - Use markdown such as headers (starting with ##), code blocks, bullet
          points, indentation for sub bullets, and backticks for inline code.
        - Don't start the response with a markdown header.
        - Don't say things like "according to the provided context".
        - CITATION REQUIREMENT: For every specific factual claim (numbers, dates,
          names, statistics, policy details), append an inline citation in the
          format 【来源：filename 第X页】. If no matching passage exists in the
          provided chunks, explicitly write (无引用支撑) after that claim.
        - NEGATIVE SPACE RULE: If the retrieved document chunks do NOT contain
          information directly relevant to the question, you MUST state
          "This specific information is not found in the provided documents"
          rather than answering from general knowledge. This rule is strict for
          yes/no questions and factual questions about specific policy documents.
          Do NOT fabricate paragraph numbers, page numbers, or statistics.
        - If you are unsure about something not in the documents, say so explicitly.
    """)
    directive = TRANSLATIONS[lang]["lang_directive"]
    if directive:
        base = base.rstrip() + f"\n- {directive}\n"
    return base


TaskInfo = namedtuple("TaskInfo", ["name", "function", "args"])
TaskResult = namedtuple("TaskResult", ["name", "result"])


def history_to_text(chat_history):
    return "\n".join(f"[{h['role']}]: {h['content']}" for h in chat_history)


def generate_chat_summary(messages, llm_cfg):
    prompt = build_prompt(
        instructions="Summarize this conversation as concisely as possible.",
        conversation=history_to_text(messages),
    )
    return call_llm_once(prompt, llm_cfg)


def _extract_years_from_query(query):
    tokens = query.replace("/", " ").replace("-", " ").split()
    years = []
    for token in tokens:
        if token.isdigit() and len(token) == 4 and token.startswith(("19", "20")):
            years.append(token)
    return set(years)


# -----------------------------------------------------------------------------
# Step 2: Document-signal routing
# Detects which specific policy documents a query is asking about by checking
# co-occurrence of year numbers and document-type keywords anywhere in the query.

import re as _re

_PA_YEAR_MAP = {
    "2019": "2019_PolicyAddress.pdf",
    "2020": "2020_PolicyAddress.pdf",
    "2021": "2021_PolicyAddress.pdf",
    "2022": "2022_PolicyAddress.pdf",
    "2023": "2023_PolicyAddress.pdf",
    "2024": "2024_PolicyAddress.pdf",
    "2025": "2025_PolicyAddress.pdf",
}

# Fiscal year string → budget filename  (e.g. "2025/26" → "budget2025.pdf")
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

_PA_KEYWORDS     = ["施政報告", "施政报告", "Policy Address"]
_BUDGET_KEYWORDS = ["預算案", "预算案", "财政预算", "財政預算", "Budget"]


def _detect_target_docs(query: str) -> list:
    """Returns deduplicated list of document filenames co-signalled in the query.

    Uses co-occurrence detection: a document is targeted when the query
    contains both a year number AND the corresponding document-type keyword,
    regardless of their relative positions in the sentence.
    """
    found = []

    is_pa     = any(kw in query for kw in _PA_KEYWORDS)
    is_budget = any(kw in query for kw in _BUDGET_KEYWORDS)

    if is_pa:
        for year in _re.findall(r"20\d\d", query):
            fn = _PA_YEAR_MAP.get(year)
            if fn and fn not in found:
                found.append(fn)

    if is_budget:
        # Prefer exact fiscal-year strings (e.g. "2025/26")
        fiscal_hits = _re.findall(r"20\d\d/\d\d", query)
        for fy in fiscal_hits:
            fn = _BUDGET_FISCAL_MAP.get(fy)
            if fn and fn not in found:
                found.append(fn)
        if not fiscal_hits:
            # Fallback: plain years → match fiscal years starting with that year
            for year in _re.findall(r"20\d\d", query):
                for fy, fn in _BUDGET_FISCAL_MAP.items():
                    if fy.startswith(year) and fn not in found:
                        found.append(fn)

    return found


# -----------------------------------------------------------------------------
# Low-level FAISS helpers

def _embed_query_vector(query: str, embedding_cfg: dict) -> np.ndarray:
    vec = embed_query(query, embedding_cfg)
    v = np.array([vec], dtype=np.float32)
    faiss.normalize_L2(v)
    return v


def _build_retrieved_list(distances, indices_arr, chunks_data, metadata):
    retrieved = []
    for rank, idx in enumerate(indices_arr[0]):
        if idx < 0:
            continue
        chunk = chunks_data[idx]
        md = metadata[idx] if idx < len(metadata) else {}
        retrieved.append({
            "page_content": chunk.get("page_content", "").strip(),
            "metadata": md,
            "_score": float(distances[0][rank]),
            "_rank": rank,
        })
    return retrieved


def _faiss_search_full(search_query: str, embedding_cfg: dict,
                       index, metadata: list, chunks_data: list, k: int) -> list:
    """Full-corpus FAISS similarity search."""
    qv = _embed_query_vector(search_query, embedding_cfg)
    k = min(k, index.ntotal)
    distances, indices_arr = index.search(qv, k)
    return _build_retrieved_list(distances, indices_arr, chunks_data, metadata)


def _faiss_search_filtered(search_query: str, target_doc: str,
                           embedding_cfg: dict, index, metadata: list,
                           chunks_data: list, k: int) -> list:
    """FAISS search restricted to chunks from target_doc.

    Searches a wider pool of candidates (up to full index), then keeps only
    those belonging to target_doc.  Falls back to full-corpus if the filtered
    result set is too thin.
    """
    target_indices = {i for i, md in enumerate(metadata)
                      if md.get("filename") == target_doc}
    if not target_indices:
        return _faiss_search_full(search_query, embedding_cfg, index, metadata, chunks_data, k)

    # Search wide enough to collect k hits from target_doc
    search_k = min(index.ntotal, max(k * 10, len(target_indices)))
    qv = _embed_query_vector(search_query, embedding_cfg)
    distances, indices_arr = index.search(qv, search_k)

    retrieved = []
    for rank, idx in enumerate(indices_arr[0]):
        if idx < 0:
            continue
        if idx in target_indices:
            chunk = chunks_data[idx]
            md = metadata[idx]
            retrieved.append({
                "page_content": chunk.get("page_content", "").strip(),
                "metadata": md,
                "_score": float(distances[0][rank]),
                "_rank": rank,
            })
            if len(retrieved) >= k:
                break

    # Fall back if we couldn't collect a meaningful set from target_doc
    if len(retrieved) < max(2, k // 3):
        return _faiss_search_full(search_query, embedding_cfg, index, metadata, chunks_data, k)
    return retrieved


# -----------------------------------------------------------------------------
# Step 3: Query decomposition for cross-document questions

def _decompose_cross_doc_query(query: str, target_docs: list, llm_cfg: dict) -> list:
    """Uses LLM to split a multi-doc query into per-document sub-queries.

    Returns a list of dicts: [{"target_doc": "...", "subquery": "..."}, ...]
    Falls back to the original query for each doc if parsing fails.
    """
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
        response = call_llm_once(prompt, llm_cfg)
        match = _re.search(r'\[.*\]', response, _re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            if isinstance(parsed, list) and all("target_doc" in x and "subquery" in x for x in parsed):
                return parsed
    except Exception:
        pass
    # Fallback: use original query for each doc
    return [{"target_doc": d, "subquery": query} for d in target_docs]


# -----------------------------------------------------------------------------
# Step 4: HyDE — Hypothetical Document Embeddings for English queries

def _is_english_query(query: str) -> bool:
    """Returns True when the query is predominantly ASCII (English)."""
    if not query:
        return False
    stripped = [c for c in query if not c.isspace()]
    if not stripped:
        return False
    ascii_ratio = sum(1 for c in stripped if ord(c) < 128) / len(stripped)
    return ascii_ratio > 0.6


def _hyde_expand_query(query: str, llm_cfg: dict) -> str:
    """Generates a short hypothetical Chinese answer to use as the retrieval query.

    This bridges the semantic gap when the corpus is Chinese but the query is English.
    """
    prompt = (
        "请用1-2句简体中文简要回答以下关于香港政策的问题。"
        "如不确定，可根据常识猜测，重点包含政策相关关键词。\n"
        f"问题：{query}"
    )
    try:
        return call_llm_once(prompt, llm_cfg)
    except Exception:
        return query  # silently fall back to original on error


# -----------------------------------------------------------------------------
# Step 5: Query Rewrite
# Rewrites the user's natural-language question into a retrieval-optimised
# keyword query before embedding.  Works for both Chinese and English inputs;
# for English queries it runs first, then HyDE further expands the rewritten
# text into a hypothetical Chinese answer.

def _rewrite_for_retrieval(query: str, llm_cfg: dict) -> str:
    """Converts a conversational question into a keyword-dense retrieval query.

    Strips interrogative framing, expands key entities, preserves numbers,
    years, and proper nouns — all of which are critical for policy recall.
    Falls back to the original query on any error.
    """
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
        result = call_llm_once(prompt, llm_cfg).strip()
        # Sanity-check: if LLM returned something very long or empty, fall back
        if result and len(result) < len(query) * 3:
            return result
    except Exception:
        pass
    return query


# -----------------------------------------------------------------------------
# Step 6: LLM-based Reranking
# After FAISS retrieves a larger candidate pool (k × pool_factor), the LLM
# scores every (query, chunk) pair in a single batch call and re-orders them.
# This is far more accurate than cosine similarity alone because the LLM can
# evaluate whether a chunk actually *answers* the question, not just whether
# it is topically related.

def _llm_rerank(query: str, chunks: list, top_k: int, llm_cfg: dict) -> list:
    """Re-ranks retrieved chunks by answer-relevance using a single LLM call.

    Sends all candidates in one prompt; parses JSON scores; returns top_k.
    Falls back to the original order on parse failure.
    """
    if not chunks or len(chunks) <= top_k:
        return chunks[:top_k]

    # Truncate each chunk to keep the prompt manageable
    snippet_len = max(200, 1200 // len(chunks))
    lines = [
        f"[{i + 1}] {c['page_content'][:snippet_len]}"
        for i, c in enumerate(chunks)
    ]
    candidates_text = "\n\n".join(lines)

    prompt = textwrap.dedent(f"""
        你是信息检索评估助手。请评估以下每个文档片段与问题的相关性，
        判断标准：该片段是否包含能直接回答问题的信息（数字、事实、政策细节）。

        问题：{query}

        文档片段：
        {candidates_text}

        请输出JSON数组，每个元素包含 idx（1起始）和 score（0-10整数，10=完全匹配）：
        [{{"idx": 1, "score": 8}}, {{"idx": 2, "score": 3}}, ...]
        仅输出JSON，不要任何解释。
    """)
    try:
        response = call_llm_once(prompt, llm_cfg)
        match = _re.search(r'\[.*\]', response, _re.DOTALL)
        if match:
            scores_data = json.loads(match.group())
            score_map = {
                int(s["idx"]) - 1: float(s["score"])
                for s in scores_data
                if isinstance(s, dict) and "idx" in s and "score" in s
            }
            scored = sorted(
                enumerate(chunks),
                key=lambda x: score_map.get(x[0], 0.0),
                reverse=True,
            )
            return [c for _, c in scored[:top_k]]
    except Exception:
        pass
    return chunks[:top_k]


# -----------------------------------------------------------------------------
# Main retrieval function (orchestrates Steps 2–6)

def search_relevant_docs(
    query: str,
    embedding_model_key: str,
    rag_cfg: dict,
    llm_cfg: dict | None = None,
    query_rewrite: bool = False,
    reranker: bool = False,
    pool_factor: int = 3,
):
    """Retrieves relevant chunks using FAISS with the configured RAG strategy.

    Pipeline order:
      Step 5 — Query Rewrite: convert question → keyword query (all queries,
                               if enabled; applied before HyDE).
      Step 4 — HyDE: English queries are expanded into a hypothetical Chinese
                     answer (applied after rewrite, replaces search_query).
      Step 2 — Routing: queries mentioning a specific document are searched
                        within that document's chunks only.
      Step 3 — Decomposition: multi-doc queries are split by LLM into
                               per-document sub-queries, each retrieved
                               independently, then merged.
      Step 6 — Reranking: candidates retrieved at k×pool_factor are re-scored
                          by LLM for answer-relevance; top-k kept.
    """
    if not rag_cfg["use_rag"]:
        return ""

    index, metadata = get_rag_index(embedding_model_key)
    chunks_data = get_all_chunks_data()
    if index is None or not chunks_data:
        return ""

    embedding_cfg = EMBEDDING_MODELS[embedding_model_key]
    k = rag_cfg.get("k", 10)
    # Inflate candidate pool when reranker is active
    fetch_k = k * pool_factor if (reranker and llm_cfg) else k

    # Step 5: Query Rewrite — before HyDE so HyDE can further expand keywords
    search_query = query
    if query_rewrite and llm_cfg:
        search_query = _rewrite_for_retrieval(query, llm_cfg)

    # Step 4: HyDE — replace search query for English inputs
    if llm_cfg and _is_english_query(search_query):
        search_query = _hyde_expand_query(search_query, llm_cfg)

    # Step 2 + 3: document routing (always against original query for signal detection)
    target_docs = _detect_target_docs(query)

    if len(target_docs) >= 2 and llm_cfg:
        # Step 3: decompose into per-doc sub-queries
        sub_queries = _decompose_cross_doc_query(query, target_docs, llm_cfg)
        sub_k = max(3, fetch_k // max(len(sub_queries), 1))

        seen = set()
        retrieved = []
        for sq in sub_queries:
            results = _faiss_search_filtered(
                sq["subquery"], sq["target_doc"],
                embedding_cfg, index, metadata, chunks_data, sub_k,
            )
            for r in results:
                key = hash(r["page_content"][:120])
                if key not in seen:
                    seen.add(key)
                    retrieved.append(r)

        # Top-up from full corpus if not enough distinct chunks
        if len(retrieved) < fetch_k:
            for r in _faiss_search_full(search_query, embedding_cfg, index, metadata, chunks_data, fetch_k):
                key = hash(r["page_content"][:120])
                if key not in seen:
                    seen.add(key)
                    retrieved.append(r)
                    if len(retrieved) >= fetch_k:
                        break

    elif len(target_docs) == 1:
        # Step 2: single-doc filtering
        retrieved = _faiss_search_filtered(
            search_query, target_docs[0],
            embedding_cfg, index, metadata, chunks_data, fetch_k,
        )

    else:
        # No document signal → full corpus search
        retrieved = _faiss_search_full(search_query, embedding_cfg, index, metadata, chunks_data, fetch_k)

    # Step 6: LLM Reranking — score candidates by answer-relevance, keep top-k
    if reranker and llm_cfg and len(retrieved) > k:
        retrieved = _llm_rerank(query, retrieved, k, llm_cfg)

    # --- Post-processing (unchanged from original) ---
    if rag_cfg.get("filter_by_year"):
        retrieved = sorted(
            retrieved,
            key=lambda d: d["metadata"].get("year", "0") or "0",
            reverse=True,
        )

    if rag_cfg.get("merge_chunks"):
        retrieved = merge_adjacent_chunks(retrieved)

    context_lines = []
    for i, doc in enumerate(retrieved):
        md = doc.get("metadata", {})
        context_lines.append(
            "\\n".join([
                f"[Chunk {i + 1}]",
                f"source_type={md.get('source_type', 'Unknown')}; year={md.get('year', 'Unknown')}; "
                f"file={md.get('filename', 'Unknown')}; page={md.get('page', 'Unknown')}; "
                f"score={doc.get('_score', 0):.4f}",
                doc.get("page_content", "").strip(),
            ])
        )
    return "\n\n".join(context_lines)


def search_extra_context(query, rag_cfg):
    """Applies lightweight metadata filtering for targeted policy queries."""
    if not rag_cfg["use_rag"]:
        return ""

    chunks_data = get_all_chunks_data()
    if not chunks_data:
        return ""

    query_lower = query.lower()
    years = _extract_years_from_query(query)

    wanted_source = None
    if "budget" in query_lower or "預算" in query or "预算" in query:
        wanted_source = "Budget"
    elif "policy address" in query_lower or "施政報告" in query or "施政报告" in query:
        wanted_source = "Policy Address"

    matched = []
    for item in chunks_data:
        md = item.get("metadata", {})
        source_type = md.get("source_type", "")
        year = str(md.get("year", ""))
        if wanted_source and source_type != wanted_source:
            continue
        if years and year not in years:
            continue
        matched.append(item)
        if len(matched) >= EXTRA_CONTEXT_LEN:
            break

    if not matched:
        return ""

    lines = []
    for i, item in enumerate(matched, start=1):
        md = item.get("metadata", {})
        lines.append(
            "\\n".join([
                f"[Filtered {i}]",
                f"source_type={md.get('source_type', 'Unknown')}; year={md.get('year', 'Unknown')}; "
                f"file={md.get('filename', 'Unknown')}; page={md.get('page', 'Unknown')}",
                item.get("page_content", "").strip(),
            ])
        )
    return "\n\n".join(lines)


def build_question_prompt(
    question,
    llm_cfg,
    embedding_model_key,
    rag_cfg,
    lang,
    query_rewrite: bool = False,
    reranker: bool = False,
    pool_factor: int = 3,
):
    """Fetches info from different sources and creates the prompt string."""
    old_history = st.session_state.messages[:-HISTORY_LENGTH]
    recent_history = st.session_state.messages[-HISTORY_LENGTH:]
    recent_history_str = history_to_text(recent_history) if recent_history else None

    task_infos = []
    if SUMMARIZE_OLD_HISTORY and old_history:
        task_infos.append(
            TaskInfo("old_message_summary", generate_chat_summary, (old_history, llm_cfg))
        )
    if rag_cfg["use_rag"] and DOCS_CONTEXT_LEN:
        task_infos.append(
            TaskInfo(
                "policy_documents",
                search_relevant_docs,
                (question, embedding_model_key, rag_cfg, llm_cfg,
                 query_rewrite, reranker, pool_factor),
            )
        )
    if rag_cfg["use_rag"] and EXTRA_CONTEXT_LEN:
        task_infos.append(
            TaskInfo("extra_context", search_extra_context, (question, rag_cfg))
        )

    results = executor.map(
        lambda ti: TaskResult(name=ti.name, result=ti.function(*ti.args)),
        task_infos,
    )
    context = {name: result for name, result in results}
    context = {k: v for k, v in context.items() if v}

    prompt = build_prompt(
        instructions=get_instructions(lang),
        **context,
        recent_messages=recent_history_str,
        question=question,
    )
    return prompt, context


# -----------------------------------------------------------------------------
# Feedback & Telemetry

def send_telemetry(**kwargs):
    pass


def show_feedback_controls(message_index):
    st.write("")
    with st.popover(t["feedback_btn"]):
        with st.form(key=f"feedback-{message_index}", border=False):
            with st.container(gap=None):
                st.markdown(f":small[{t['feedback_rating']}]")
                st.feedback(options="stars")
            st.text_area(t["feedback_detail"])
            if st.checkbox(t["feedback_history"], True):
                pass
            ""
            if st.form_submit_button(t["feedback_submit"]):
                pass


@st.dialog(t["disclaimer_title"])
def show_disclaimer_dialog():
    st.caption(t["disclaimer_text"])


# -----------------------------------------------------------------------------
# Rest of Sidebar (after language selection)

with st.sidebar:
    st.header(t["sidebar_header"])

    st.subheader(t["llm_section"])
    selected_llm_key = st.selectbox(
        t["llm_section"],
        options=list(LLM_MODELS.keys()),
        format_func=lambda k: LLM_MODEL_LABELS[selected_lang][k],
        index=0,
        label_visibility="collapsed",
    )
    llm_cfg = LLM_MODELS[selected_llm_key]

    if llm_cfg["provider"] == "xai":
        grok_key = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
        if not grok_key:
            st.warning(t["xai_warning"], icon="⚠️")

    st.divider()

    st.subheader(t["rag_section"])
    selected_rag_key = st.selectbox(
        t["rag_section"],
        options=list(RAG_CONFIGS.keys()),
        format_func=lambda k: RAG_CONFIG_LABELS[selected_lang][k],
        index=0,
        label_visibility="collapsed",
    )
    rag_cfg = RAG_CONFIGS[selected_rag_key].copy()
    st.caption(RAG_CONFIG_DESCRIPTIONS[selected_lang][selected_rag_key])

    if rag_cfg["use_rag"]:
        rag_cfg["k"] = st.slider(
            t["rag_k_label"],
            min_value=3,
            max_value=20,
            value=rag_cfg.get("k", 10),
            step=1,
        )
        rag_cfg["merge_chunks"] = st.toggle(
            t["rag_merge_label"],
            value=rag_cfg.get("merge_chunks", True),
        )
        rag_cfg["filter_by_year"] = st.toggle(
            t["rag_sort_label"],
            value=rag_cfg.get("filter_by_year", False),
        )

    st.divider()

    st.subheader(t["embedding_section"])
    selected_embedding_key = st.selectbox(
        t["embedding_section"],
        options=list(EMBEDDING_MODELS.keys()),
        format_func=lambda k: EMBEDDING_MODEL_LABELS[selected_lang][k],
        index=0,
        label_visibility="collapsed",
    )
    embedding_cfg = EMBEDDING_MODELS[selected_embedding_key]

    if embedding_cfg["provider"] == "huggingface":
        st.info(t["hf_info"], icon="ℹ️")

    _, index_file, _ = _get_index_paths(embedding_cfg)
    if index_file.exists():
        st.success(t["index_ready"], icon="✅")
    else:
        st.warning(t["index_missing"], icon="⏳")

    st.divider()

    # ---- Retrieval Optimisation controls ----
    st.subheader(t["retrieval_opt_section"])

    rag_query_rewrite = st.toggle(t["query_rewrite_label"], value=False)
    rag_reranker = st.toggle(t["reranker_label"], value=False)

    if rag_reranker:
        rag_pool_factor = st.slider(
            t["pool_factor_label"],
            min_value=2,
            max_value=5,
            value=3,
            help=t["pool_factor_help"],
        )
    else:
        rag_pool_factor = 3

    st.divider()

    st.subheader(t["metrics_section"])
    st.caption(t["metrics_caption"])
    cols = st.columns(2)
    with cols[0]:
        st.metric("RAG-Basic", "43%")
        st.metric("RAG-Opt.", "37%")
    with cols[1]:
        st.metric("RAG-Basic", "2.97/6")
        st.metric("RAG-Opt.", "2.90/6")
    st.caption(t["metrics_note"])


# -----------------------------------------------------------------------------
# Main UI

st.html(div(style=styles(font_size=rem(5), line_height=1))["🇭🇰"])

title_row = st.container(horizontal=True, vertical_alignment="bottom")

with title_row:
    st.title(t["page_title"], anchor=False, width="stretch")

user_just_asked_initial_question = (
    "initial_question" in st.session_state and st.session_state.initial_question
)
user_just_clicked_suggestion = (
    "selected_suggestion" in st.session_state and st.session_state.selected_suggestion
)
user_first_interaction = user_just_asked_initial_question or user_just_clicked_suggestion

has_message_history = (
    "messages" in st.session_state and len(st.session_state.messages) > 0
)

if not user_first_interaction and not has_message_history:
    st.session_state.messages = []

    with st.container():
        st.chat_input(t["chat_placeholder"], key="initial_question")

        selected_suggestion = st.pills(
            label="Examples",
            label_visibility="collapsed",
            options=suggestions.keys(),
            key="selected_suggestion",
        )

    st.button(
        t["disclaimer_btn"],
        type="tertiary",
        on_click=show_disclaimer_dialog,
    )

    st.stop()

user_message = st.chat_input(t["chat_followup"])

if not user_message:
    if user_just_asked_initial_question:
        user_message = st.session_state.initial_question
    if user_just_clicked_suggestion:
        user_message = suggestions[st.session_state.selected_suggestion]

with title_row:

    def clear_conversation():
        st.session_state.messages = []
        st.session_state.initial_question = None
        st.session_state.selected_suggestion = None

    st.button(
        t["restart_button"],
        icon=":material/refresh:",
        on_click=clear_conversation,
    )

if "prev_question_timestamp" not in st.session_state:
    st.session_state.prev_question_timestamp = datetime.datetime.fromtimestamp(0)

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.container()
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_feedback_controls(i)

if user_message:
    user_message = user_message.replace("$", r"\$")

    with st.chat_message("user"):
        st.text(user_message)

    with st.chat_message("assistant"):
        with st.spinner(t["spinner_waiting"]):
            question_timestamp = datetime.datetime.now()
            time_diff = question_timestamp - st.session_state.prev_question_timestamp
            st.session_state.prev_question_timestamp = question_timestamp
            if time_diff < MIN_TIME_BETWEEN_REQUESTS:
                time.sleep(time_diff.seconds + time_diff.microseconds * 0.001)
            user_message = user_message.replace("'", "")

        rag_display = RAG_CONFIG_LABELS[selected_lang][selected_rag_key]
        llm_display = LLM_MODEL_LABELS[selected_lang][selected_llm_key]
        config_label = f"{llm_display} · {rag_display} · {EMBEDDING_MODEL_LABELS[selected_lang][selected_embedding_key]}"

        if DEBUG_MODE:
            with st.status(t["spinner_prompt"]) as status:
                full_prompt, retrieved_context = build_question_prompt(
                    user_message, llm_cfg, selected_embedding_key, rag_cfg, selected_lang,
                    query_rewrite=rag_query_rewrite,
                    reranker=rag_reranker,
                    pool_factor=rag_pool_factor,
                )
                st.code(full_prompt)
                status.update(label=t["spinner_prompt_done"])
        else:
            with st.spinner(t["spinner_researching"].format(rag=rag_display)):
                full_prompt, retrieved_context = build_question_prompt(
                    user_message, llm_cfg, selected_embedding_key, rag_cfg, selected_lang,
                    query_rewrite=rag_query_rewrite,
                    reranker=rag_reranker,
                    pool_factor=rag_pool_factor,
                )

        with st.spinner(t["spinner_thinking"].format(llm=llm_display)):
            response_gen = call_llm_stream(full_prompt, llm_cfg)

        with st.container():
            response = st.write_stream(response_gen)

            retrieved_docs = retrieved_context.get("policy_documents", "")
            if retrieved_docs:
                with st.expander(t["chunks_expander"], expanded=False):
                    st.text(retrieved_docs)

            st.caption(f":gray[{config_label}]")

            st.session_state.messages.append({"role": "user", "content": user_message})
            st.session_state.messages.append({"role": "assistant", "content": response})

            show_feedback_controls(len(st.session_state.messages) - 1)
            send_telemetry(question=user_message, response=response)

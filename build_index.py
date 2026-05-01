"""
构建 FAISS 向量索引脚本。

用法：
    uv run python build_index.py                          # 默认 OpenAI text-embedding-3-small
    uv run python build_index.py --model openai-large     # OpenAI text-embedding-3-large
    uv run python build_index.py --model bge-small-zh     # BAAI/bge-small-zh-v1.5（开源）
    uv run python build_index.py --model bge-m3           # BAAI/bge-m3（开源多语言）
    uv run python build_index.py --batch-size 200         # 自定义批大小
    uv run python build_index.py --force                  # 强制重建已存在的索引
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PROCESSED_DIR = PROJECT_DIR / "data" / "data_processed"
INDEX_BASE_DIR = PROJECT_DIR / ".rag_index"

CHUNK_FILES = [
    "hk_policy_chunks.json",
    "legco_hansard_chunks.json",
    "Public_Open_Space_chunks.json",
    "Public_Transport_Nodes_chunks.json",
]

# ---------------------------------------------------------------------------
# Embedding model registry (same as streamlit_app.py)

EMBEDDING_MODELS = {
    "openai-small": {
        "provider": "openai",
        "model_id": "text-embedding-3-small",
        "index_subdir": "openai_small",
        "description": "OpenAI text-embedding-3-small (1536维)",
    },
    "openai-large": {
        "provider": "openai",
        "model_id": "text-embedding-3-large",
        "index_subdir": "openai_large",
        "description": "OpenAI text-embedding-3-large (3072维)",
    },
    "bge-small-zh": {
        "provider": "huggingface",
        "model_id": "BAAI/bge-small-zh-v1.5",
        "index_subdir": "hf_bge_small_zh",
        "description": "BAAI/bge-small-zh-v1.5 开源中文小模型 (512维)",
    },
    "bge-m3": {
        "provider": "huggingface",
        "model_id": "BAAI/bge-m3",
        "index_subdir": "hf_bge_m3",
        "description": "BAAI/bge-m3 开源多语言模型 (1024维)",
    },
}

DEFAULT_MODEL = "openai-small"


# ---------------------------------------------------------------------------
# Data loading

def load_all_chunks():
    all_chunks = []
    for fname in CHUNK_FILES:
        path = DATA_PROCESSED_DIR / fname
        if not path.exists():
            print(f"  [跳过] 未找到: {path}")
            continue
        print(f"  加载: {fname}", end="", flush=True)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        count = 0
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
            count += 1
        print(f"  → {count} 条")
    return all_chunks


# ---------------------------------------------------------------------------
# Embedding helpers

def embed_batch_openai(texts, model_id, client):
    response = client.embeddings.create(model=model_id, input=texts)
    return [item.embedding for item in response.data]


def embed_batch_huggingface(texts, hf_model):
    return hf_model.embed_documents(texts)


# ---------------------------------------------------------------------------
# Index building

def build_index(model_key, batch_size, force):
    if model_key not in EMBEDDING_MODELS:
        print(f"错误：未知模型 '{model_key}'，可选值：{list(EMBEDDING_MODELS.keys())}")
        sys.exit(1)

    cfg = EMBEDDING_MODELS[model_key]
    index_dir = INDEX_BASE_DIR / cfg["index_subdir"]
    index_file = index_dir / "index.faiss"
    metadata_file = index_dir / "metadata.json"

    print(f"\n{'='*60}")
    print(f"Embedding 模型: {cfg['description']}")
    print(f"索引目录:       {index_dir}")
    print(f"{'='*60}\n")

    if index_file.exists() and metadata_file.exists() and not force:
        print("索引已存在。使用 --force 强制重建。")
        print(f"  index.faiss  : {index_file.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"  metadata.json: {metadata_file.stat().st_size / 1024:.1f} KB")
        return

    # Load data
    print("加载文档块...")
    chunks = load_all_chunks()
    if not chunks:
        print("错误：未找到任何文档块，请检查 data/data_processed/ 目录。")
        sys.exit(1)
    print(f"共加载 {len(chunks)} 条文档块\n")

    texts = [c["page_content"] for c in chunks]
    metadata = [c["metadata"] for c in chunks]

    # Initialize embedding model
    print("初始化 Embedding 模型...")
    if cfg["provider"] == "openai":
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("错误：未设置 OPENAI_API_KEY 环境变量。")
            sys.exit(1)
        client = OpenAI(api_key=api_key)
        embed_fn = lambda batch: embed_batch_openai(batch, cfg["model_id"], client)
        print(f"  使用 OpenAI API ({cfg['model_id']})")

    elif cfg["provider"] == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        hf_model = HuggingFaceEmbeddings(model_name=cfg["model_id"])
        embed_fn = lambda batch: embed_batch_huggingface(batch, hf_model)
        print(f"  使用 HuggingFace 本地模型 ({cfg['model_id']})")

    else:
        print(f"错误：未知 provider '{cfg['provider']}'")
        sys.exit(1)

    # Embed in batches
    print(f"\n开始 Embedding（批大小={batch_size}）...")
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    t_start = time.time()

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        batch_num = i // batch_size + 1
        elapsed = time.time() - t_start
        eta = (elapsed / batch_num) * (total_batches - batch_num) if batch_num > 1 else 0
        print(
            f"  批次 {batch_num:4d}/{total_batches}  "
            f"({i + len(batch):6d}/{len(texts)})  "
            f"已用 {elapsed:5.1f}s  预计剩余 {eta:5.1f}s",
            end="\r",
            flush=True,
        )
        vecs = embed_fn(batch)
        all_embeddings.extend(vecs)

    print(f"\n  Embedding 完成，共耗时 {time.time() - t_start:.1f}s")

    # Build FAISS index
    print("\n构建 FAISS 索引...")
    vectors = np.array(all_embeddings, dtype=np.float32)
    faiss.normalize_L2(vectors)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    print(f"  向量维度: {dim}，总向量数: {index.ntotal}")

    # Persist
    index_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n保存索引到 {index_dir} ...")
    faiss.write_index(index, str(index_file))
    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    size_mb = index_file.stat().st_size / 1024 / 1024
    print(f"  index.faiss  : {size_mb:.1f} MB")
    print(f"  metadata.json: {metadata_file.stat().st_size / 1024:.1f} KB")
    print(f"\n✅ 索引构建完成！")


# ---------------------------------------------------------------------------
# CLI

def main():
    parser = argparse.ArgumentParser(
        description="构建 HK Policy RAG 的 FAISS 向量索引",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n可用模型：\n"
        + "\n".join(f"  {k:16s}  {v['description']}" for k, v in EMBEDDING_MODELS.items()),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=list(EMBEDDING_MODELS.keys()),
        help=f"Embedding 模型（默认：{DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="每批嵌入的文档数（默认：100）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重建已存在的索引",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用模型及索引状态",
    )

    args = parser.parse_args()

    if args.list:
        print("\n可用 Embedding 模型：\n")
        for key, cfg in EMBEDDING_MODELS.items():
            index_file = INDEX_BASE_DIR / cfg["index_subdir"] / "index.faiss"
            status = "✅ 已构建" if index_file.exists() else "⬜ 未构建"
            print(f"  {key:16s}  {status}  {cfg['description']}")
        print()
        return

    build_index(model_key=args.model, batch_size=args.batch_size, force=args.force)


if __name__ == "__main__":
    main()

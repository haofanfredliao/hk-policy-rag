import json
from pathlib import Path
import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings  # 新版路径
from langchain_community.vectorstores import FAISS

DATA_FOLDER = r"D:\ljc\7307\data_processed"
INDEX_PATH = r"D:\ljc\7307\hk_policy_index"

files = [
    "hk_policy_chunks.json",
    "legco_hansard_chunks.json",
    "Public_Open_Space_chunks.json",
    "Public_Transport_Nodes_chunks.json"
]

# 加载所有文档
all_docs = []
for fname in files:
    path = Path(DATA_FOLDER) / fname
    if not path.exists():
        print(f"跳过: {fname}")
        continue
    print(f"正在加载: {fname}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        text = item.get("page_content") or item.get("content")
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
        all_docs.append(Document(page_content=text, metadata=meta))

print(f"总共加载了 {len(all_docs)} 个文档块")

print("正在初始化小模型 BAAI/bge-small-zh-v1.5 ...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 分批构建 FAISS（每批 2000）
batch_size = 2000
print(f"开始分批构建 FAISS 索引，每批 {batch_size} 条...")
vectorstore = None
for i in range(0, len(all_docs), batch_size):
    batch = all_docs[i:i+batch_size]
    print(f"处理第 {i//batch_size + 1} 批，共 {len(batch)} 条")
    if vectorstore is None:
        vectorstore = FAISS.from_documents(batch, embeddings)
    else:
        vectorstore.add_documents(batch)
    print(f"  当前索引包含 {vectorstore.index.ntotal} 条")

print("正在保存索引...")
vectorstore.save_local(INDEX_PATH)
print(f"✅ 索引构建完成，保存至 {INDEX_PATH}")
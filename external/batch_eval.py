import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from xai_sdk import Client
from xai_sdk.chat import system, user
import time

# ========== 配置 ==========
INDEX_PATH = r"D:\ljc\7307\hk_policy_index"  # 当前使用的索引（small模型）
# 如果想用 base 模型，改成 hk_policy_index_base
QUESTIONS_PATH = r"D:\ljc\7307\test_questions.json"
OUTPUT_EXCEL = r"D:\ljc\7307\results_raw_v3.xlsx"  # 新版本输出
GROK_API_KEY = "..."  # 替换成真实的 Grok API Key

# 三种配置定义（k=10，RAG 配置增加合并相邻 chunk）
CONFIGS = {
    "Baseline": {
        "use_rag": False,
        "k": 0,
        "description": "直接问 LLM，无检索"
    },
    "RAG-Basic": {
        "use_rag": True,
        "k": 10,
        "merge_chunks": True,  # 新增：合并相邻 chunk
        "description": "RAG 基础配置，k=10，合并相邻 chunk"
    },
    "RAG-Optimized": {
        "use_rag": True,
        "k": 10,
        "merge_chunks": True,
        "filter_by_year": True,  # 按年份降序排序
        "description": "RAG 优化配置，k=10，合并相邻 chunk，按年份过滤"
    }
}

# 初始化 Embedding 和 FAISS
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# 初始化 Grok 客户端
client = Client(api_key=GROK_API_KEY)


def merge_adjacent_chunks(docs):
    """
    合并同一文档中页码相邻的 chunk。
    输入：List[Document]（来自检索结果）
    输出：List[Document]（合并后的文档，page_content 为合并后的文本，metadata 合并页码范围）
    """
    if not docs:
        return docs

    # 按文件名分组
    grouped = defaultdict(list)
    for doc in docs:
        filename = doc.metadata.get("filename", "unknown")
        grouped[filename].append(doc)

    merged_docs = []
    for filename, doc_list in grouped.items():
        # 按页码排序
        doc_list.sort(key=lambda d: d.metadata.get("page", 0) if d.metadata.get("page") is not None else 0)

        merged_blocks = []
        current_block = None
        for doc in doc_list:
            page = doc.metadata.get("page")
            # 如果 page 是 None，无法合并，单独作为一个块
            if page is None:
                merged_blocks.append(doc)
                continue

            if current_block is None:
                current_block = {
                    "docs": [doc],
                    "start_page": page,
                    "end_page": page,
                    "texts": [doc.page_content]
                }
            else:
                # 检查是否相邻（页码差 <= 1）
                if page - current_block["end_page"] <= 1:
                    current_block["docs"].append(doc)
                    current_block["end_page"] = page
                    current_block["texts"].append(doc.page_content)
                else:
                    # 完成当前块
                    merged_text = "\n\n".join(current_block["texts"])
                    merged_meta = current_block["docs"][0].metadata.copy()
                    merged_meta["start_page"] = current_block["start_page"]
                    merged_meta["end_page"] = current_block["end_page"]
                    merged_meta["page"] = f"{current_block['start_page']}-{current_block['end_page']}"
                    merged_doc = type(doc)(
                        page_content=merged_text,
                        metadata=merged_meta
                    )
                    merged_blocks.append(merged_doc)
                    # 开始新块
                    current_block = {
                        "docs": [doc],
                        "start_page": page,
                        "end_page": page,
                        "texts": [doc.page_content]
                    }
        # 处理最后一个块
        if current_block:
            merged_text = "\n\n".join(current_block["texts"])
            merged_meta = current_block["docs"][0].metadata.copy()
            merged_meta["start_page"] = current_block["start_page"]
            merged_meta["end_page"] = current_block["end_page"]
            merged_meta["page"] = f"{current_block['start_page']}-{current_block['end_page']}"
            merged_doc = type(doc)(
                page_content=merged_text,
                metadata=merged_meta
            )
            merged_blocks.append(merged_doc)

        merged_docs.extend(merged_blocks)

    return merged_docs


def ask_grok(question, config):
    """根据配置调用 Grok 生成答案"""
    if not config["use_rag"]:
        prompt = f"你是一个香港城市政策问答助手。请回答以下问题：\n\n{question}"
        # 调用 Grok
        chat = client.chat.create(model="grok-4-1-fast-non-reasoning")
        chat.append(system("你是专业、准确的香港政策问答助手。"))
        chat.append(user(prompt))
        response = chat.sample()
        answer = response.content
        sources = []
    else:
        # RAG: 检索相关文档
        k = config["k"]
        docs = vectorstore.similarity_search(question, k=k)

        # 按年份降序排序（如果启用）
        if config.get("filter_by_year"):
            docs = sorted(docs, key=lambda d: d.metadata.get("year", "0"), reverse=True)

        # 合并相邻 chunk（如果启用）
        if config.get("merge_chunks"):
            docs = merge_adjacent_chunks(docs)

        # 构建上下文
        context_parts = []
        for idx, doc in enumerate(docs):
            filename = doc.metadata.get("filename", "未知")
            page_info = doc.metadata.get("page", "无页码")
            # 如果 page 是范围（如"1-3"），直接显示
            context_parts.append(f"【来源：{filename} 第{page_info}页】\n{doc.page_content}")
        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""你是一个香港城市政策问答助手。请根据以下参考文段回答用户的问题。

**重要指示**：
1. 优先使用参考文段中的信息回答。如果参考文段中有明确答案，请直接引用并标注来源（文件名和页码）。
2. 如果参考文段中没有相关信息，或者信息不完整，你可以结合你自己的知识进行补充，但**必须明确说明**哪些内容不是来自参考文段（例如：“根据我的知识，...”）。
3. 不要编造虚假信息。如果你不确定，请如实说“不确定”。

参考文段：
{context}

用户问题：{question}

回答："""

        # 调用 Grok
        chat = client.chat.create(model="grok-4-1-fast-non-reasoning")
        chat.append(system("你是专业、准确的香港政策问答助手。"))
        chat.append(user(prompt))
        response = chat.sample()
        answer = response.content

        # 提取引用来源
        sources = []
        for doc in docs:
            filename = doc.metadata.get("filename", "未知")
            page_info = doc.metadata.get("page", "无页码")
            sources.append(f"{filename} 第{page_info}页")

    return answer, sources


def main():
    # 加载测试题
    with open(QUESTIONS_PATH, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    if isinstance(questions_data, dict):
        for v in questions_data.values():
            if isinstance(v, list):
                questions_data = v
                break

    results = []
    for q_item in questions_data:
        q_id = q_item.get("id", q_item.get("question_id", "unknown"))
        question_text = q_item.get("question", q_item.get("text", ""))
        if not question_text:
            continue
        print(f"处理问题 {q_id}: {question_text[:50]}...")

        for config_name, config in CONFIGS.items():
            print(f"  配置: {config_name}")
            try:
                answer, sources = ask_grok(question_text, config)
                result_row = {
                    "question_id": q_id,
                    "question": question_text,
                    "config": config_name,
                    "answer": answer,
                    "sources": "; ".join(sources) if sources else "",
                }
                results.append(result_row)
            except Exception as e:
                print(f"    错误: {e}")
                result_row = {
                    "question_id": q_id,
                    "question": question_text,
                    "config": config_name,
                    "answer": f"ERROR: {e}",
                    "sources": "",
                }
                results.append(result_row)
            time.sleep(0.5)  # 避免 API 速率限制

    # 保存为 Excel
    df = pd.DataFrame(results)
    df = df.sort_values(["question_id", "config"])
    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"✅ 批量实验完成，结果保存至 {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()
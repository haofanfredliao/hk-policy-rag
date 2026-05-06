import json
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from xai_sdk import Client
from xai_sdk.chat import system, user
import time

load_dotenv()

# ========== 配置 ==========
PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_EXCEL = PROJECT_DIR / "data" / "result" / "results_raw_v4.xlsx"
GROUND_TRUTH_JSON = PROJECT_DIR / "docs" / "ground_truth.json"
SCORING_RUBRIC_MD = PROJECT_DIR / "docs" / "scoring_rubric.md"
OUTPUT_SCORED_EXCEL = PROJECT_DIR / "data" / "result" / "results_scored_v4.xlsx"
OUTPUT_METRICS_JSON = PROJECT_DIR / "data" / "result" / "final_metrics.json"
GROK_API_KEY = os.getenv("GROK_API_KEY", "")

# 裁判模型使用的评分 prompt（你可以根据 rubric.md 修改）
DEFAULT_SCORING_PROMPT = """
你是一个公正的评分裁判。请根据以下评分标准对模型答案进行评分。

评分标准：
- 正确性 (0-3分)：
    * 3分：答案与标准答案完全一致（数字/事实完全匹配，表述方式可以不同）。
    * 2分：答案与标准答案基本一致，但有小偏差（如单位、约数等）。
    * 1分：答案部分正确，但关键信息错误或缺失。
    * 0分：答案完全错误或与标准答案无关。
- 幻觉 (0-2分，扣分制)：
    * 2分：完全没有幻觉，答案完全基于标准答案或真实知识，无编造。
    * 1分：有轻微幻觉（编造了细节但不影响主要答案）。
    * 0分：严重幻觉（编造了关键信息或整个答案虚假）。
- 引用准确性（仅针对RAG配置，Baseline配置此项自动给0分）：
    * 1分：引用的文件名/页码正确匹配标准答案的来源。
    * 0分：引用不准确或未提供引用。

最终总分 = 正确性分 + 幻觉分 + 引用分（Baseline中引用分固定为0）。总分范围 0-6 分。

请输出JSON格式，只包含以下字段：
{
    "correctness": 整数,
    "hallucination": 整数,
    "citation": 整数,
    "total_score": 整数,
    "reason": "简短理由"
}
"""

# 初始化 Grok 客户端
if not GROK_API_KEY:
    raise ValueError("GROK_API_KEY not set. Add it to your .env file.")
client = Client(api_key=GROK_API_KEY)


def load_ground_truth(gt_path):
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 假设 data 是列表，每个元素有 id, ground_truth, gt_source 等
    gt_dict = {}
    for item in data:
        qid = item.get("id")
        if qid:
            gt_dict[qid] = {
                "ground_truth": item.get("ground_truth", ""),
                "gt_source": item.get("gt_source", ""),
                "source_doc": item.get("source_doc", "")
            }
    return gt_dict


def score_with_grok(question, model_answer, ground_truth, gt_source, config_name):
    """调用 Grok 作为裁判打分"""
    # 构建裁判 prompt
    if config_name == "Baseline":
        prompt = f"""问题：{question}
模型答案：{model_answer}
标准答案：{ground_truth}
标准答案来源：{gt_source}
配置：Baseline（无检索，无引用）

{DEFAULT_SCORING_PROMPT}
注意：Baseline配置没有引用，因此引用项自动给0分。请只评价正确性和幻觉。输出JSON格式。"""
    else:
        # RAG 配置，模型答案中可能包含引用
        prompt = f"""问题：{question}
模型答案：{model_answer}
标准答案：{ground_truth}
标准答案来源：{gt_source}
配置：{config_name}（有检索，答案中可能包含引用）

{DEFAULT_SCORING_PROMPT}
请评估答案的正确性、幻觉，以及引用是否准确匹配标准答案来源（如果答案没有引用，引用分为0）。输出JSON格式。"""

    chat = client.chat.create(model="grok-4-1-fast-non-reasoning")
    chat.append(system("你是一个严格、公正的评分裁判。"))
    chat.append(user(prompt))
    response = chat.sample()
    # 解析 JSON 响应
    try:
        # 尝试提取JSON部分
        text = response.content.strip()
        # 寻找第一个 { 和最后一个 }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            json_str = text[start:end + 1]
            score_data = json.loads(json_str)
        else:
            raise ValueError("No JSON found")
        return score_data
    except Exception as e:
        print(f"评分响应解析失败: {e}")
        print(f"原始响应: {response.content}")
        # 返回默认
        return {"correctness": 0, "hallucination": 0, "citation": 0, "total_score": 0, "reason": "解析失败"}


def main():
    # 1. 加载 ground truth
    gt = load_ground_truth(GROUND_TRUTH_JSON)
    print(f"加载了 {len(gt)} 条标准答案")

    # 2. 读取实验结果
    df = pd.read_excel(INPUT_EXCEL)
    # 确保列名正确
    required_cols = ['question_id', 'question', 'config', 'answer', 'sources']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Excel 缺少列: {col}")

    # 3. 逐行评分
    scores = []
    for idx, row in df.iterrows():
        qid = row['question_id']
        question = row['question']
        config = row['config']
        answer = row['answer']
        sources = row['sources'] if pd.notna(row['sources']) else ""

        if qid not in gt:
            print(f"警告: 未找到问题 {qid} 的标准答案，跳过")
            continue

        ground_truth_text = gt[qid]['ground_truth']
        gt_source_text = gt[qid]['gt_source']

        print(f"评分: {qid} | {config}")
        score_info = score_with_grok(question, answer, ground_truth_text, gt_source_text, config)

        scores.append({
            "question_id": qid,
            "question": question,
            "config": config,
            "answer": answer,
            "sources": sources,
            "ground_truth": ground_truth_text,
            "correctness": score_info.get("correctness", 0),
            "hallucination": score_info.get("hallucination", 0),
            "citation": score_info.get("citation", 0),
            "total_score": score_info.get("total_score", 0),
            "reason": score_info.get("reason", "")
        })
        time.sleep(0.5)  # 避免 API 限流

    # 4. 保存打分后的 Excel
    df_scores = pd.DataFrame(scores)
    df_scores.to_excel(OUTPUT_SCORED_EXCEL, index=False)
    print(f"打分结果保存至 {OUTPUT_SCORED_EXCEL}")

    # 5. 计算汇总指标
    # 按配置分组统计
    metrics = {}
    for config in df_scores['config'].unique():
        sub = df_scores[df_scores['config'] == config]
        avg_correctness = sub['correctness'].mean()
        avg_hallucination = sub['hallucination'].mean()
        avg_citation = sub['citation'].mean() if config != "Baseline" else 0
        avg_total = sub['total_score'].mean()
        # 准确率：总分>=4 视为正确
        accuracy = (sub['total_score'] >= 4).mean()
        # 幻觉率：幻觉分<2 的比例（或定义严格）
        hallucination_rate = (sub['hallucination'] < 2).mean()

        metrics[config] = {
            "avg_correctness": round(avg_correctness, 2),
            "avg_hallucination": round(avg_hallucination, 2),
            "avg_citation": round(avg_citation, 2),
            "avg_total_score": round(avg_total, 2),
            "accuracy": round(accuracy, 2),
            "hallucination_free_rate": round(hallucination_rate, 2)
        }

    # 总体指标
    total_avg = df_scores['total_score'].mean()
    total_accuracy = (df_scores['total_score'] >= 4).mean()

    final_metrics = {
        "overall": {
            "avg_total_score": round(total_avg, 2),
            "accuracy": round(total_accuracy, 2)
        },
        "by_config": metrics
    }

    with open(OUTPUT_METRICS_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)
    print(f"汇总指标保存至 {OUTPUT_METRICS_JSON}")
    print(json.dumps(final_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

PROCESSED_DIR = "data_processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 同样执行组长要求的 600/100 标准
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, chunk_overlap=100, separators=["\n\n", "\n", "。", "！"]
)

def find_geojson_in_folder(folder_name):
    """地毯式搜索：只要文件夹里有 .geojson，不管藏多深都找出来"""
    for root, dirs, files in os.walk(folder_name):
        for file in files:
            if file.endswith(".geojson"):
                return os.path.join(root, file)
    return None

def process_geojson_folder(folder_name, dataset_label):
    print(f"\n🗺️ 正在掃描文件夾: {folder_name}")
    geojson_path = find_geojson_in_folder(folder_name)
    
    if not geojson_path:
        print(f"❌ 警告：在 {folder_name} 中沒找到 .geojson 文件！")
        return

    print(f"📄 成功鎖定文件: {geojson_path}")
    
    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        all_text = []
        # 将空间属性转化为自然语言
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            props_desc = "，".join([f"{k}為{v}" for k,v in props.items() if v])
            all_text.append(f"關於【{dataset_label}】的空間節點記錄：{props_desc}。")
        
        # 文本切块
        chunks = text_splitter.split_text("\n\n".join(all_text))
        
        # 注入 Metadata
        final_data = [{"content": c, "metadata": {"source": dataset_label, "type": "Spatial"}} for c in chunks]
        
        output_file = os.path.join(PROCESSED_DIR, f"{dataset_label}_chunks.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 【{dataset_label}】處理成功！生成了 {len(final_data)} 個文本塊！")
    except Exception as e:
        print(f"❌ 轉換失敗: {e}")

if __name__ == "__main__":
    # 严格对应你电脑上的两个真实文件夹名
    folder_transport = "PublicTransportFacilitiesandBusInterchangesDesignatedasNoSmokingAreas_GEOJSON"
    folder_openspace = "PublicOpenSpace_GEOJSON"
    
    process_geojson_folder(folder_transport, "Transport_Nodes")
    process_geojson_folder(folder_openspace, "Open_Spaces")
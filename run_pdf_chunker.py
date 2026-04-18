import sys
import subprocess
import os
import json

# ==========================================
# 第一步：自动环境配置 (修复了新版 LangChain 依赖)
# ==========================================
def install_requirements():
    print(">>> 正在检查运行环境...")
    # 新增了 langchain-text-splitters
    packages = ["langchain", "langchain-community", "langchain-text-splitters", "pymupdf"]
    for pkg in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])
            print(f" [OK] {pkg} 准备就绪")
        except Exception as e:
            print(f" [Error] 自动安装 {pkg} 失败: {e}")

install_requirements()

# ==========================================
# 第二步：导入核心工具 (修复了新版路径)
# ==========================================
print("\n>>> 环境检测完毕，正在加载 AI 处理工具...")
try:
    from langchain_community.document_loaders import PyMuPDFLoader
    # 路径更新为新版：langchain_text_splitters
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError as e:
    print(f"\n[致命错误] 库导入失败，请检查网络后重试。\n详细信息: {e}")
    input("按回车键退出...")
    sys.exit()

# ==========================================
# 第三步：定义路径和分块器
# ==========================================
paths = {
    "Policy Address": r"C:\Users\22833\Desktop\7307\小组\Policy_PDFs\PolicyAddress",
    "Budget": r"C:\Users\22833\Desktop\7307\小组\Policy_PDFs\Budget"
}

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", ".", " ", ""]
)

# ==========================================
# 第四步：执行 PDF 批量解析与切分
# ==========================================
all_chunks = []
print("\n>>> 开始处理 PDF 文件...")

for category, dir_path in paths.items():
    if not os.path.exists(dir_path):
        print(f" [警告] 找不到路径: {dir_path}")
        continue
        
    for filename in os.listdir(dir_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(dir_path, filename)
            
            year = "".join(filter(str.isdigit, filename))[:4] 
            if not year:
                year = "Unknown"
            
            print(f" 正在解析: {filename} (提取年份: {year})")
            
            try:
                loader = PyMuPDFLoader(file_path)
                pages = loader.load()
                
                for page in pages:
                    page.metadata["source_type"] = category
                    page.metadata["year"] = year
                    page.metadata["filename"] = filename
                
                chunks = text_splitter.split_documents(pages)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f" [错误] 处理 {filename} 时发生异常: {e}")

print(f"\n>>> PDF 解析完成！共生成 {len(all_chunks)} 个 Chunk。")

# ==========================================
# 第五步：保存为 JSON 格式
# ==========================================
print("\n>>> 正在将数据打包存入 JSON...")
chunks_data = []
for chunk in all_chunks:
    chunks_data.append({
        "page_content": chunk.page_content,
        "metadata": chunk.metadata
    })

output_filepath = r"C:\Users\22833\Desktop\7307\小组\hk_policy_chunks.json"

try:
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    print(f"\n🎉 大功告成！数据已成功保存至：\n{output_filepath}")
except Exception as e:
    print(f"\n[保存错误] 写入 JSON 文件时失败: {e}")

print("-" * 50)
input("任务结束！请按回车键 (Enter) 关闭此窗口...")

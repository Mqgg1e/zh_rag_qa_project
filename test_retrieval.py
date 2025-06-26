from models.embedder import load_embedder
from models.retriever import load_retriever
import torch
def main():
    query = "这部动画适合小学生看吗？"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 正在加载嵌入模型，使用设备: {device}")

    print("🔧 正在加载嵌入模型...")
    embedder = load_embedder(device=device)
    print("📦 正在加载向量索引...")
    retriever = load_retriever()

    print("🔍 正在向量化查询并检索...")
    query_vec = embedder.encode([query])
    docs = retriever.search(query_vec, top_k=3)

    print(f"\n📄 检索结果：")
    for i, doc in enumerate(docs, 1):
        print(f"\n== 第 {i} 条 ==")
        print(doc)

if __name__ == "__main__":
    main()

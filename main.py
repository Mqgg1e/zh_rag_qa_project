import yaml
from models.embedder import Embedder
from models.retriever import Retriever
from pipeline.rag_pipeline import RAGPipeline

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    cfg = load_config()
    embedder = Embedder(cfg["embed_model"], device=cfg["device"])
    retriever = Retriever(cfg["faiss_index_path"], cfg["faiss_csv_path"])
    rag = RAGPipeline(embedder, retriever, cfg["llm_model"], cfg["device"])

    while True:
        query = input("请输入问题（输入exit退出）：")
        if query.lower() == "exit":
            break
        answer = rag.run(query, top_k=cfg["top_k"], max_new_tokens=cfg["max_new_tokens"], temperature=cfg["temperature"])
        print("回答：", answer)

import yaml
from models.embedder import Embedder
from models.retriever import Retriever
from pipeline.rag_pipeline import RAGPipeline
import torch

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    cfg = load_config()

    def resolve_device(device_str):
        if device_str == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_str

    embedder_device = resolve_device(cfg["embedder"]["device"])
    llm_device = resolve_device(cfg["llm"]["device"])

    embedder = Embedder(cfg["embedder"]["model_name"], device=embedder_device)
    retriever = Retriever(cfg["index"]["path"], cfg["index"]["docs"])
    rag = RAGPipeline(embedder, retriever, cfg["llm"]["model_name"], device=llm_device)

    while True:
        query = input("请输入问题（输入exit退出）：")
        if query.lower() == "exit":
            break
        answer = rag.run(
            query,
            top_k=cfg.get("top_k", 3),
            max_new_tokens=cfg.get("max_new_tokens", 200),
            temperature=cfg.get("temperature", 0.7)
        )
        print("回答：", answer)

from sentence_transformers import CrossEncoder
import torch

def resolve_device(device_str):
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str
class Reranker:
    def __init__(self, model_name: str, device: str = "auto"):
        self.device = resolve_device(device)
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(self, query: str, passages: list[str], top_k: int = 3) -> list[str]:
        pairs = [[query, passage] for passage in passages]
        scores = self.model.predict(pairs)
        scored_passages = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored_passages[:top_k]]
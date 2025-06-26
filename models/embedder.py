from sentence_transformers import SentenceTransformer
import torch

def resolve_device(device_str):
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str

class Embedder:
    def __init__(self, model_name: str, device: str = "auto"):
        self.device = resolve_device(device)
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str], batch_size: int = 32) -> list:
        if not isinstance(texts, list):
            raise ValueError("`texts` 应为字符串列表")
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)

def load_embedder(model_name="BAAI/bge-large-zh-v1.5", device="auto") -> Embedder:
    return Embedder(model_name, device)

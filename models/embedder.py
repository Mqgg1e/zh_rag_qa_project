from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str]) -> list:
        return self.model.encode(texts, show_progress_bar=False)

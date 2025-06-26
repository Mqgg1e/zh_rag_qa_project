import faiss
import pandas as pd

class Retriever:
    def __init__(self, index_path: str, corpus_path: str):
        self.index = faiss.read_index(index_path)
        df = pd.read_csv(corpus_path)
        self.corpus = df["clean_text"].tolist()

    def search(self, query_vec, top_k: int = 3) -> list[str]:
        scores, indices = self.index.search(query_vec.astype("float32"), top_k)
        return [self.corpus[i] for i in indices[0]]

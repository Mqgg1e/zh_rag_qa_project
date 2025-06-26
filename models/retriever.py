import faiss
import pandas as pd

class Retriever:
    def __init__(self, index_path: str, corpus_path: str):
        self.index = faiss.read_index(index_path)
        df = pd.read_csv(corpus_path)
        self.corpus = df["clean_text"].tolist()

    def search(self, query_vec, top_k: int = 3, return_scores: bool = False):
        scores, indices = self.index.search(query_vec.astype("float32"), top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.corpus):
                if return_scores:
                    results.append((self.corpus[idx], scores[0][i]))
                else:
                    results.append(self.corpus[idx])
        return results

def load_retriever(index_path="vector_index/faiss_index.index", corpus_path="vector_index/faiss_docs.csv") -> Retriever:
    return Retriever(index_path, corpus_path)

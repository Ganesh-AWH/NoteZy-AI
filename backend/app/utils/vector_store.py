import faiss
import numpy as np

class FAISSVectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.texts = []

    def add(self, embeddings: np.ndarray, texts: list[str]):
        self.index.add(embeddings)
        self.texts.extend(texts)

    def search(self, query_embedding: np.ndarray, k=3):
        scores, indices = self.index.search(query_embedding, k)
        results = []

        for i, idx in enumerate(indices[0]):
            results.append({
                "score": float(scores[0][i]),
                "text": self.texts[idx]
            })

        return results

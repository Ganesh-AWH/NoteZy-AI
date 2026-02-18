import faiss
import numpy as np
import pickle
from pathlib import Path


class FAISSVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.texts = []
        self.files = set()   # NEW

    # ===============================
    # ADD VECTORS
    # ===============================
    def add(self, embeddings: np.ndarray, texts: list[str]):
        self.index.add(embeddings)
        self.texts.extend(texts)

    # ===============================
    # SEARCH
    # ===============================
    def search(self, query_embedding: np.ndarray, k=3):
        scores, indices = self.index.search(query_embedding, k)
        results = []

        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            results.append({
                "score": float(scores[0][i]),
                "text": self.texts[idx]
            })

        return results

    # ===============================
    # SAVE / LOAD
    # ===============================
    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "index.faiss"))

        with open(path / "texts.pkl", "wb") as f:
            pickle.dump(self.texts, f)

        with open(path / "files.pkl", "wb") as f:
            pickle.dump(self.files, f)

    @classmethod
    def load(cls, path: str):
        path = Path(path)

        index = faiss.read_index(str(path / "index.faiss"))

        with open(path / "texts.pkl", "rb") as f:
            texts = pickle.load(f)

        with open(path / "files.pkl", "rb") as f:
            files = pickle.load(f)

        store = cls(index.d)
        store.index = index
        store.texts = texts
        store.files = files
        return store


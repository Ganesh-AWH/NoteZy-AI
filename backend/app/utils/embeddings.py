import numpy as np
from sentence_transformers import SentenceTransformer
"""
Sentence library is maintained on Hugging Face
and used by SentenceTransformers.

"""

class EmbeddingModel:
    # model_name = SentenceTransformer("all-MiniLM-L6-v2") -> (later we can switch to all-mpnet-base-v2 for higher quality.)
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True #It improves cosine similarity results
        )
        

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.reshape(1, -1)
# print(embeddings.shape) #->(3, 384) 3 chunks Each chunk → 384-dimensional vector




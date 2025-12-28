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



#testing
# chunks = [
#     "Machine learning enables computers to learn patterns from data without being explicitly programmed.",
#     "Embeddings convert text into numerical vectors that capture semantic meaning.",
#     "Vector databases allow efficient similarity search over high-dimensional embeddings.",
#     "Photosynthesis is the process by which green plants convert sunlight into chemical energy."
# ]

# from numpy import dot
# from numpy.linalg import norm

# embedder = EmbeddingModel()
# embeddings = embedder.embed_texts(chunks)

# def cosine_similarity(a, b):
#     return dot(a, b) / (norm(a) * norm(b))

# query = "How do embeddings represent the meaning of text?"

# query_embedding = embedder.embed_query(query)
# print(embeddings.shape)
# print(query_embedding.shape)


# max_score = 0
# chunk_index = None
# for i, emb in enumerate(embeddings):
#     score = cosine_similarity(query_embedding, emb)
#     print(f"Chunk {i+1} similarity: {score:.3f}")
    
#     if score > max_score:
#         max_score = score
#         chunk_index = i        
    
# print(chunks[chunk_index])
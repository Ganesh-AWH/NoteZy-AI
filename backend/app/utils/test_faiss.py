from chunker import WordChunker
from embeddings import EmbeddingModel
from vector_store import FAISSVectorStore


def manual_faiss_test():
    # -----------------------------
    # 1️⃣ Test text (intentionally mixed topics)
    # -----------------------------
    text = """
    Machine learning enables computers to learn from data without being explicitly programmed.
    Embeddings convert text into numerical vectors that capture semantic meaning.
    FAISS is a library used for fast similarity search over vector embeddings.
    Photosynthesis is the process by which plants make food using sunlight.
    """

    # -----------------------------
    # 2️⃣ Initialize components
    # -----------------------------
    chunker = WordChunker(chunk_size=10, overlap=3)
    embedder = EmbeddingModel()

    # -----------------------------
    # 3️⃣ Chunk the text
    # -----------------------------
    chunks = chunker.chunk(text)

    print("\n=== CHUNKS ===")
    for i, c in enumerate(chunks):
        print(f"{i}: {c}")

    # -----------------------------
    # 4️⃣ Generate embeddings
    # -----------------------------
    embeddings = embedder.embed_texts(chunks)

    print("\nEmbedding shape:", embeddings.shape)

    # -----------------------------
    # 5️⃣ Create FAISS index
    # -----------------------------
    vector_store = FAISSVectorStore(dim=embeddings.shape[1])
    vector_store.add(embeddings, chunks)

    print("\nFAISS index size:", len(vector_store.texts))

    # -----------------------------
    # 6️⃣ Query test
    # -----------------------------
    query = "What is FAISS used for?"

    query_embedding = embedder.embed_query(query)

    results = vector_store.search(query_embedding, k=3)

    # -----------------------------
    # 7️⃣ Print results
    # -----------------------------
    print("\n=== SEARCH RESULTS ===")
    for rank, r in enumerate(results, 1):
        print(f"\nRank {rank}")
        print(f"Score: {r['score']:.4f}")
        print(f"Text: {r['text']}")


if __name__ == "__main__":
    manual_faiss_test()
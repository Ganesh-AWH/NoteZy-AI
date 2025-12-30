from utils.file_loader import FileTextExtractor
from utils.text_cleaner import RandomCaseTextCleaner
from utils.chunker import WordChunker
from utils.embeddings import EmbeddingModel
from utils.vector_store import FAISSVectorStore


class UtilsServicePipeline:
    """
    End-to-end pipeline:
    File → Clean → Chunk → Embed → Store → Search
    """

    def __init__(self, chunk_size=40, overlap=10):
        self.extractor = FileTextExtractor()
        self.cleaner_cls = RandomCaseTextCleaner
        self.chunker = WordChunker(chunk_size, overlap)
        self.embedder = EmbeddingModel()
        self.vector_store = None

    # ===============================
    # INGESTION
    # ===============================
    def ingest(self, file_path: str):
        # 1️⃣ Extract text
        raw_text = self.extractor.extract_text(file_path)
        # 2️⃣ Clean text
        cleaned_text = self.cleaner_cls(raw_text).clean()
        # 3️⃣ Chunk text
        chunks = self.chunker.chunk(cleaned_text)
        if not chunks:
            raise ValueError("No chunks generated from document")

        # 4️⃣ Embeddings
        embeddings = self.embedder.embed_texts(chunks)

        # 5️⃣ Vector store
        dim = embeddings.shape[1]
        self.vector_store = FAISSVectorStore(dim)
        self.vector_store.add(embeddings, chunks)

        return {
            "status": "success",
            "chunks": len(chunks),
            "embedding_dim": dim
        }

    # ===============================
    # SEARCH
    # ===============================
    def search(self, query: str, k=3):
        if self.vector_store is None:
            raise RuntimeError("Pipeline not ingested yet")

        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.search(query_embedding, k)
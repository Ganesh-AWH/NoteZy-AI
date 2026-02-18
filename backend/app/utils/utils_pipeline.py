from utils.file_loader import FileTextExtractor
from utils.text_cleaner import RandomCaseTextCleaner
from utils.chunker import WordChunker
from utils.embeddings import EmbeddingModel
from utils.vector_store import FAISSVectorStore
from pathlib import Path


class UtilsServicePipeline:
    def __init__(self, chunk_size=40, overlap=10, store_path="vector_store"):
        self.extractor = FileTextExtractor()
        self.cleaner_cls = RandomCaseTextCleaner
        self.chunker = WordChunker(chunk_size, overlap)
        self.embedder = EmbeddingModel()
        self.store_path = store_path

        # Auto-load if exists
        if Path(store_path).exists():
            self.vector_store = FAISSVectorStore.load(store_path)
        else:
            self.vector_store = None

    # ===============================
    # INGEST
    # ===============================
    def ingest(self, file_path: str):
        file_name = Path(file_path).name

        if self.vector_store and file_name in self.vector_store.files:
            return {
                "status": "skipped",
                "reason": "File already ingested"
            }

        raw_text = self.extractor.extract_text(file_path)
        cleaned_text = self.cleaner_cls(raw_text).clean()
        chunks = self.chunker.chunk(cleaned_text)
        embeddings = self.embedder.embed_texts(chunks)

        if self.vector_store is None:
            self.vector_store = FAISSVectorStore(embeddings.shape[1])

        self.vector_store.add(embeddings, chunks)
        self.vector_store.files.add(file_name)
        self.vector_store.save(self.store_path)

        return {
            "status": "success",
            "chunks_added": len(chunks),
        }


    # ===============================
    # SEARCH
    # ===============================
    def search(self, query: str, k=3):
        if self.vector_store is None:
            raise RuntimeError("No vector store found. Ingest documents first.")

        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.search(query_embedding, k)

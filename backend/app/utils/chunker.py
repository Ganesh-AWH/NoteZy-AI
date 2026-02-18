import re
from typing import List

class WordChunker:
    """
    Robust word-based chunker that works for ANY text,
    even without punctuation.
    """

    def __init__(self, chunk_size=40, overlap=10):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        # Normalize text
        text = re.sub(r"\s+", " ", text.strip())
        words = text.split()

        chunks = []
        start = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))
            start = end - self.overlap  # overlap works ALWAYS

        return chunks


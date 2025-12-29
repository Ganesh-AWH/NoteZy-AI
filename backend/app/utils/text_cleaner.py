import re


class RandomCaseTextCleaner:
    """
    Cleans text with random capitalization into normal sentence case.
    Suitable for RAG / NLP pipelines.
    """

    def __init__(self, text: str):
        self.text = text
#Whitespace
    def normalize_whitespace(self):
        self.text = re.sub(r"\s+", " ", self.text).strip()
        return self
#lowercase
    def to_lowercase(self):
        self.text = self.text.lower()
        return self
#split_sentences
    def split_sentences(self):
        self.sentences = re.split(r"(?<=[.!?])\s+|\n+", self.text)
        return self
#to_sentence_case
    def to_sentence_case(self):
        cleaned = []
        for s in self.sentences:
            s = s.strip()
            if s:
                cleaned.append(s[0].upper() + s[1:])
        self.text = "\n".join(cleaned)
        return self

    def clean(self) -> str:
        """
        Executes full cleaning pipeline
        """
        return (
            self.normalize_whitespace()
            .to_lowercase()
            .split_sentences()
            .to_sentence_case()
            .text
        )
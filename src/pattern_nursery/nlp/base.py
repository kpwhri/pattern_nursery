from abc import ABC, abstractmethod
from typing import List, Tuple


class NLPEngine(ABC):
    @abstractmethod
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        pass

    @abstractmethod
    def split_sentences_with_spans(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into sentences with char spans."""
        pass

    @abstractmethod
    def tokenize(self, text: str, lower: bool = False) -> List[str]:
        """Tokenize text."""
        pass

    @abstractmethod
    def tokenize_with_spans(self, text: str) -> List[Tuple[str, int, int]]:
        """Tokenize text with char spans."""
        pass

    @abstractmethod
    def noun_chunks(self, text: str) -> List[str]:
        """Extract noun chunks from text."""
        pass

    def ngrams(self, text: str, n: int = 2, lower: bool = False) -> List[str]:
        """Generate n-grams from text."""
        tokens = self.tokenize(text, lower=lower)
        if len(tokens) < n:
            return []
        return [' '.join(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]

    @property
    def name(self) -> str:
        """Return the name of the NLP engine."""
        return self.__class__.__name__

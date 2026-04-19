from typing import List, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from pattern_nursery.embeddings.base import EmbeddingBackend


class TfidfBackend(EmbeddingBackend):
    """tfidf sparse embedding backend using scikit-learn's TfidfVectorizer as a proxy or rank_bm25."""

    def __init__(self, corpus: List[str] = None):
        self.vectorizer = TfidfVectorizer()
        if corpus:
            self.vectorizer.fit(corpus)

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        # TODO: this should probably be sparse
        return self.vectorizer.transform(texts).toarray().astype(np.float32)

    @property
    def dimension(self) -> int:
        if hasattr(self.vectorizer, 'vocabulary_'):
            return len(self.vectorizer.vocabulary_)
        return 0

    @property
    def model_name(self) -> str:
        return 'tfidf'

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Union

import numpy as np

from pattern_nursery.embeddings.base import EmbeddingBackend

_TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


class BM25Backend(EmbeddingBackend):
    """
    BM25 document/query vector backend.

    This backend builds a BM25-weighted vector space over a fitted corpus.
    It is not a classic search-engine BM25 scorer; instead it exposes BM25
    term weights as dense vectors so it can plug into existing vector stores
    through `EmbeddingBackend`.

    Notes:
    - `embed()` returns BM25-weighted vectors.
    - The vector dimension is the fitted vocabulary size.
    - For best results, call `fit(corpus)` or pass a corpus to `__init__`.
    """

    def __init__(
            self,
            corpus: Optional[List[str]] = None,
            *,
            k1: float = 1.5,
            b: float = 0.75,
            lowercase: bool = True,
            min_df: int = 1,
    ):
        self.k1 = float(k1)
        self.b = float(b)
        self.lowercase = lowercase
        self.min_df = int(min_df)

        self._vocab: Dict[str, int] = {}
        self._idf: np.ndarray = np.zeros(0, dtype=np.float32)
        self._avgdl: float = 0.0
        self._fitted: bool = False

        if corpus is not None:
            self.fit(corpus)

    def fit(self, corpus: List[str]) -> "BM25Backend":
        """
        Fit vocabulary, document frequencies, IDF, and average document length.
        """
        if not corpus:
            self._vocab = {}
            self._idf = np.zeros(0, dtype=np.float32)
            self._avgdl = 0.0
            self._fitted = True
            return self

        doc_freqs: Counter[str] = Counter()
        doc_lengths: List[int] = []

        tokenized_docs: List[List[str]] = []
        for text in corpus:
            tokens = self._tokenize(text)
            tokenized_docs.append(tokens)
            doc_lengths.append(len(tokens))
            doc_freqs.update(set(tokens))

        # Filter vocabulary by min_df
        vocab_terms = [term for term, df in doc_freqs.items() if df >= self.min_df]
        vocab_terms.sort()
        self._vocab = {term: idx for idx, term in enumerate(vocab_terms)}

        n_docs = len(corpus)
        self._avgdl = float(sum(doc_lengths)) / n_docs if n_docs else 0.0

        idf = np.zeros(len(self._vocab), dtype=np.float32)
        for term, idx in self._vocab.items():
            df = doc_freqs[term]
            # standard bm25 idf with +1 smoothing inside the log to avoid negatives on common terms
            idf[idx] = float(math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)))

        self._idf = idf
        self._fitted = True
        return self

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        if not self._fitted:
            raise ValueError("BM25Backend is not fitted. Pass a corpus to __init__ or call fit().")

        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        if self.dimension == 0:
            return np.zeros((len(texts), 0), dtype=np.float32)

        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)

        for row, text in enumerate(texts):
            tokens = self._tokenize(text)
            if not tokens:
                continue

            tf = Counter(tok for tok in tokens if tok in self._vocab)
            if not tf:
                continue

            dl = len(tokens)
            denom_len = self.k1 * (1.0 - self.b + self.b * (dl / self._avgdl)) if self._avgdl > 0 else self.k1

            for term, freq in tf.items():
                col = self._vocab[term]
                idf = self._idf[col]
                score = idf * (freq * (self.k1 + 1.0)) / (freq + denom_len)
                vectors[row, col] = score

        return vectors

    @property
    def dimension(self) -> int:
        return len(self._vocab)

    @property
    def model_name(self) -> str:
        return "bm25"

    def _tokenize(self, text: str) -> List[str]:
        if text is None:
            return []
        if self.lowercase:
            text = text.lower()
        return _TOKEN_RE.findall(text)

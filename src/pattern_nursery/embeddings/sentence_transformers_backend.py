import time
from typing import List, Union

import numpy as np
from loguru import logger

from pattern_nursery.embeddings.base import EmbeddingBackend


class SentenceTransformersBackend(EmbeddingBackend):
    """Sentence Transformers embedding backend."""

    def __init__(self, model_path: str):
        """Model path must be local"""
        self._model_path = model_path
        try:
            from sentence_transformers import SentenceTransformer
            t0 = time.perf_counter()
            self.model = SentenceTransformer(model_path, device='cpu')
            elapsed = time.perf_counter() - t0
            logger.info(f'Initialized SBERT backend with model {model_path} in {elapsed:.3f}s')
        except ImportError:
            raise ImportError('sentence-transformers is required')

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        t0 = time.perf_counter()
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        elapsed = time.perf_counter() - t0
        logger.debug(f'SBERT embed: n_texts={len(texts)}, dim={self.dimension}, elapsed={elapsed:.3f}s')
        return embeddings

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return self._model_path

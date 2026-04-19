import time

import numpy as np
from loguru import logger

from pattern_nursery.embeddings.base import EmbeddingBackend


class MockBackend(EmbeddingBackend):
    def __init__(self, model_path: str = 'mock'):
        self._dimension = 384
        self.model_path = model_path
        logger.info(f'Initialized Mock backend with dimension {self._dimension}')

    def embed(self, texts):
        t0 = time.perf_counter()
        res = np.random.rand(len(texts), self.dimension).astype(np.float32)
        elapsed = time.perf_counter() - t0
        logger.debug(f'Mock embed: n_texts={len(texts)}, dim={self.dimension}, elapsed={elapsed:.3f}s')
        return res

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self.model_path

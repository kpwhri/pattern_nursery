from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np


class EmbeddingBackend(ABC):
    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Embed texts into vectors."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""
        pass

    @property
    def model_name(self) -> str:
        """Return the name/path of the model."""
        return 'unknown'

    def get_similar_terms(self, term: str, top_k: int = 10) -> List[dict]:
        """Get similar terms directly from the model if supported."""
        return []

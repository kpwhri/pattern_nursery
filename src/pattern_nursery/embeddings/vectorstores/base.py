from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np


class VectorStore(ABC):
    @abstractmethod
    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]):
        """Add vectors to the store."""
        pass

    @abstractmethod
    def query(self, query_vectors: np.ndarray, top_k: int = 10) -> List[List[Tuple[int, float, Dict[str, Any]]]]:
        """Query the store for similar vectors."""
        pass

    @abstractmethod
    def persist(self, path: Path):
        """Persist the store to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path, collection_name: str = None) -> 'VectorStore':
        """Load the store from disk."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return the number of vectors in the store."""
        pass

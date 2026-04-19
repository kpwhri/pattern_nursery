from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

from pattern_nursery.embeddings.vectorstores.base import VectorStore


class ChromaStore(VectorStore):
    """ChromaDB-based vector store."""

    def __init__(self, collection_name: str, persist_dir: Path):
        try:
            import chromadb
        except ImportError:
            raise ImportError('chromadb is required for ChromaStore')

        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.collection_name = collection_name
        self.persist_dir = persist_dir

    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]):
        ids = [str(i) for i in range(self.count(), self.count() + len(vectors))]
        # chromadb expects list of lists for embeddings
        self.collection.add(
            embeddings=vectors.tolist(),
            metadatas=metadata,
            ids=ids
        )

    def query(self, query_vectors: np.ndarray, top_k: int = 10) -> List[List[Tuple[int, float, Dict[str, Any]]]]:
        results = self.collection.query(
            query_embeddings=query_vectors.tolist(),
            n_results=top_k
        )

        formatted_results = []
        for i in range(len(query_vectors)):
            query_results = []
            if results['ids'][i]:
                for j in range(len(results['ids'][i])):
                    query_results.append((
                        int(results['ids'][i][j]),
                        results['distances'][i][j],
                        results['metadatas'][i][j]
                    ))
            formatted_results.append(query_results)
        return formatted_results

    def persist(self, path: Path):
        # ChromaDB PersistentClient handles persistence automatically
        pass

    @classmethod
    def load(cls, path: Path, collection_name: str = None) -> 'ChromaStore':
        return cls(collection_name=self.collection_name, persist_dir=path)

    def count(self) -> int:
        return self.collection.count()

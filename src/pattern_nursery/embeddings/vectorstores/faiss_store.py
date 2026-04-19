import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
from loguru import logger

from pattern_nursery.embeddings.vectorstores.base import VectorStore


class FaissStore(VectorStore):
    """FAISS-based vector store."""

    def __init__(self, dimension: int, index_type: str = 'flatip', nlist: int = 1024):
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError('faiss-cpu is required for FaissStore')

        self.dimension = int(dimension)
        if index_type == 'flatip':
            # inner product (use for cosine similarity if vectors are normalized)
            self.index = faiss.IndexFlatIP(self.dimension)
        elif index_type == 'l2':
            self.index = faiss.IndexFlatL2(self.dimension)
        elif index_type == 'ivfflatip':
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, int(nlist), faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f'Unsupported index type: {index_type}')

        self.metadata: List[Dict[str, Any]] = []

    def train(self, vectors: np.ndarray):
        """Train the index (required for IVF)."""
        if not self.index.is_trained:
            vectors = vectors.astype(np.float32)
            self.faiss.normalize_L2(vectors)
            self.index.train(vectors)

    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]):
        if vectors.shape[1] != self.dimension:
            raise ValueError(f'Vector dimension {vectors.shape[1]} does not match store dimension {self.dimension}')
        if len(metadata) != vectors.shape[0]:
            raise ValueError(f'Metadata count ({len(metadata)}) must match number of vectors ({vectors.shape[0]}).')

        # faiss expects float32
        vectors = vectors.astype(np.float32)
        # normalize for Inner Product to get Cosine Similarity
        self.faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.metadata.extend(metadata)

        # safety check: keep store consistent
        if self.index.ntotal != len(self.metadata):
            raise RuntimeError(
                f'Internal mismatch after add(): index.ntotal={self.index.ntotal} '
                f'but len(metadata)={len(self.metadata)}'
            )

    def query(self, query_vectors: np.ndarray, top_k: int = 10, nprobe: int = 1) -> List[
        List[Tuple[int, float, Dict[str, Any]]]]:
        t0 = time.perf_counter()
        query_vectors = query_vectors.astype(np.float32)
        self.faiss.normalize_L2(query_vectors)

        # set nprobe if it is an IVF index
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe

        distances, indices = self.index.search(query_vectors, top_k)
        elapsed = time.perf_counter() - t0
        logger.debug(
            f'FAISS query: top_k={top_k}, query_count={len(query_vectors)}, '
            f'elapsed={elapsed:.3f}s, ntotal={self.index.ntotal}'
        )

        results = []
        for i in range(len(query_vectors)):
            query_results = []
            for j in range(len(indices[i])):
                idx = indices[i][j]
                if idx == -1:
                    continue
                query_results.append((int(idx), float(distances[i][j]), self.metadata[idx]))
            results.append(query_results)
        return results

    def persist(self, path: Path):
        t0 = time.perf_counter()
        path.mkdir(parents=True, exist_ok=True)
        self.faiss.write_index(self.index, str(path / 'index.faiss'))
        with open(path / 'metadata.jsonl', 'w', encoding='utf8') as f:
            for meta in self.metadata:
                f.write(json.dumps(meta) + '\n')
        elapsed = time.perf_counter() - t0
        logger.info(f'Persisted FAISS store to {path} ({self.index.ntotal} vectors, {elapsed:.3f}s)')

    @classmethod
    def load(cls, path: Path, collection_name: str = None) -> 'FaissStore':
        """Just requires path -- collection name ignored."""
        t0 = time.perf_counter()
        import faiss
        index = faiss.read_index(str(path / 'index.faiss'))
        instance = cls(dimension=index.d)
        instance.index = index

        metadata = []
        with open(path / 'metadata.jsonl', encoding='utf8') as f:
            for line in f:
                metadata.append(json.loads(line))
        instance.metadata = metadata
        elapsed = time.perf_counter() - t0
        logger.info(f'Loaded FAISS store from {path} ({index.ntotal} vectors, {elapsed:.3f}s)')
        return instance

    def count(self) -> int:
        return self.index.ntotal

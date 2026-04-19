import numpy as np
import pytest

from pattern_nursery.embeddings.vectorstores.chroma_store import ChromaStore
from pattern_nursery.embeddings.vectorstores.faiss_store import FaissStore

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    import chromadb

    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False


@pytest.mark.skipif(not HAS_FAISS, reason='faiss-cpu not installed')
def test_faiss_store_basic(tmp_path):
    store = FaissStore(dimension=4)
    vecs = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    meta = [{'id': 1}, {'id': 2}]
    store.add(vecs, meta)
    assert store.count() == 2

    query = np.array([[1, 0, 0, 0]], dtype=np.float32)
    results = store.query(query, top_k=1)
    assert len(results) == 1
    assert results[0][0][2]['id'] == 1


@pytest.mark.skipif(HAS_FAISS, reason='testing ImportError path')
def test_faiss_import_error():
    with pytest.raises(ImportError, match='faiss-cpu is required'):
        FaissStore(4)


@pytest.mark.skipif(not HAS_CHROMA, reason='chromadb not installed')
def test_chroma_store_basic(tmp_path):
    # Chroma needs a persist_dir and collection name
    store = ChromaStore('test_coll', tmp_path)
    vecs = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    meta = [{'id': 1}, {'id': 2}]
    store.add(vecs, meta)
    assert store.count() == 2

    query = np.array([[1, 0, 0, 0]], dtype=np.float32)
    results = store.query(query, top_k=1)
    assert len(results) == 1
    assert results[0][0][2]['id'] == 1


@pytest.mark.skipif(HAS_CHROMA, reason='testing ImportError path')
def test_chroma_import_error(tmp_path):
    with pytest.raises(ImportError, match='chromadb is required'):
        ChromaStore('test', tmp_path)

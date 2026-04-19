from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pattern_nursery.index.manager import IndexManager
from pattern_nursery.nlp.regex_engine import RegexEngine


class MockBackend:
    def __init__(self, model_path='mock'):
        self.model_name = model_path
        self.dimension = 384

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return np.random.rand(len(texts), 384).astype(np.float32)


@pytest.fixture
def mock_get_backend():
    with patch('pattern_nursery.cli.main.get_embedding_backend') as mock:
        mock.side_effect = lambda backend_name, model_path: MockBackend(model_path)
        yield mock


@pytest.fixture
def mock_get_nlp():
    with patch('pattern_nursery.cli.main.get_nlp_engine') as mock:
        mock.side_effect = lambda engine_name: RegexEngine()
        yield mock


def test_parallel_indexing_notes(tmp_path, mock_get_backend, mock_get_nlp):
    work_dir = tmp_path / 'runs'
    run_name = 'parallel_test'
    manager = IndexManager(work_dir, run_name, vector_store_type='faiss', nlp_engine=RegexEngine())

    docs = [
        ('1', 'Väinämöinen, old and steadfast,'),
        ('2', 'Ground the Sampo, ground the lid-cover.'),
        ('3', 'Ilmarinen, worthy brother,'),
        ('4', 'Forged the sky and wrought the heavens.')
    ]

    # need to use a 'real' (registered) backend name for it to be recreated in workers
    backend = MockBackend(model_path='mock_model')
    backend.model_name = 'mock_model'

    def mock_backend_embed(texts):
        if isinstance(texts, str): texts = [texts]
        return np.random.rand(len(texts), 384).astype(np.float32)

    backend.embed = mock_backend_embed
    backend.__class__.__name__ = 'MockBackend'  # so it uses 'mock' name

    with patch('pattern_nursery.index.manager.IndexManager._get_or_create_store') as mock_store_factory:
        from pattern_nursery.embeddings.vectorstores.faiss_store import FaissStore
        real_store = FaissStore(384)
        mock_store_factory.return_value = real_store

        with patch('pattern_nursery.cli.main.get_embedding_backend') as mgb:
            import numpy as np
            mock_b = MagicMock()
            mock_b.dimension = 384
            mock_b.model_name = 'mock'
            mock_b.embed.side_effect = lambda texts: np.random.rand(len(texts), 384).astype(np.float32)
            mgb.return_value = mock_b

            manager.index_notes(iter(docs), mock_b, batch_size=2, num_workers=2)

    assert manager.exists('notes')
    store = manager.load_store('notes')
    assert store.count() == 4


def test_parallel_indexing_sentences(tmp_path, mock_get_backend, mock_get_nlp):
    work_dir = tmp_path / 'runs'
    run_name = 'parallel_test_sent'
    manager = IndexManager(work_dir, run_name, vector_store_type='faiss', nlp_engine=RegexEngine())

    docs = [
        ('1', 'Sentence one. Sentence two!'),
        ('2', 'Sentence three? Sentence four.')
    ]

    with patch('pattern_nursery.cli.main.get_embedding_backend') as mgb:
        import numpy as np
        mock_b = MagicMock()
        mock_b.dimension = 384
        mock_b.model_name = 'mock'
        mock_b.embed.side_effect = lambda texts: np.random.rand(len(texts), 384).astype(np.float32)
        mgb.return_value = mock_b

        with patch('pattern_nursery.cli.main.get_nlp_engine') as mne:
            mne.return_value = RegexEngine()

            manager.index_sentences(iter(docs), mock_b, batch_size=1, num_workers=2)

    assert manager.exists('sentences')
    store = manager.load_store('sentences')
    # each doc has 2 sentences
    assert store.count() == 4


def test_parallel_indexing_paragraphs(tmp_path, mock_get_backend, mock_get_nlp):
    work_dir = tmp_path / 'runs'
    run_name = 'parallel_test_para'
    manager = IndexManager(work_dir, run_name, vector_store_type='faiss', nlp_engine=RegexEngine())

    docs = [
        ('1', 'S1. S2. S3. S4. S5.')
    ]

    with patch('pattern_nursery.cli.main.get_embedding_backend') as mgb:
        import numpy as np
        mock_b = MagicMock()
        mock_b.dimension = 384
        mock_b.model_name = 'mock'
        mock_b.embed.side_effect = lambda texts: np.random.rand(len(texts), 384).astype(np.float32)
        mgb.return_value = mock_b

        with patch('pattern_nursery.cli.main.get_nlp_engine') as mne:
            mne.return_value = RegexEngine()

            # window_size=3, overlap=1 => S1-S3, S3-S5
            manager.index_paragraphs(iter(docs), mock_b, window_size=3, overlap=1, batch_size=1, num_workers=2)

    assert manager.exists('paragraphs')
    store = manager.load_store('paragraphs')
    assert store.count() == 2

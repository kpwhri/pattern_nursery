import numpy as np

from pattern_nursery.index.manager import IndexManager
from pattern_nursery.io import reader


class FakeBackend:
    def __init__(self, dimension=4):
        self.dimension = dimension

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return np.zeros((len(texts), self.dimension))


def test_build_vocabulary(tmp_workdir, regex_engine, kalevala_corpus):
    """Test vocabulary building and persistence."""
    mgr = IndexManager(tmp_workdir, 'v_run', nlp_engine=regex_engine)
    docs = reader.corpus_iter(kalevala_corpus, 'id', 'text')
    vocab = mgr.build_and_persist_vocabulary(docs, lower=False)

    assert 'Väinämöinen' in vocab
    assert 'kantele' in vocab

    vocab_file = tmp_workdir / 'runs' / 'v_run' / 'artifacts' / 'vocab.txt'
    assert vocab_file.exists()
    content = vocab_file.read_text(encoding='utf8')
    assert 'Väinämöinen' in content


def test_index_manager_config_writing(tmp_workdir, regex_engine):
    """Test manual TOML config writing."""
    mgr = IndexManager(tmp_workdir, 'c_run', nlp_engine=regex_engine)
    backend = FakeBackend(dimension=8)

    # manually trigger config write through internal method to test it
    store_path = tmp_workdir / 'runs' / 'c_run' / 'vectorstore' / 'test_coll'
    mgr._write_config('test_coll', backend.dimension, store_path, extra={'foo': 'bar'})

    config_path = store_path / 'config.toml'
    assert config_path.exists()
    text = config_path.read_text()
    assert 'dim = 8' in text
    assert 'foo = "bar"' in text


def test_index_sentences_batching(tmp_workdir, regex_engine, kalevala_corpus):
    """Test batching logic in index_sentences."""
    # mock _add_to_store to check calls
    mgr = IndexManager(tmp_workdir, 'b_run', nlp_engine=regex_engine)
    backend = FakeBackend()

    # monkeypatch _add_to_store
    call_args = []

    def mock_add(collection, backend, texts, metadata):
        call_args.append((collection, len(texts)))

    mgr._add_to_store = mock_add

    docs = reader.corpus_iter(kalevala_corpus, 'id', 'text')
    mgr.index_sentences(docs, backend)

    # 4 documents in kalevala_corpus, each has ~2 sentences
    # batch_size is 1000, so it should be 1 call
    assert len(call_args) == 1
    assert call_args[0][0] == 'sentences'
    assert call_args[0][1] > 0

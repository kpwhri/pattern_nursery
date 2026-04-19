import json

import numpy as np
from typer.testing import CliRunner

from pattern_nursery.cli.main import app
from pattern_nursery.index.manager import IndexManager

runner = CliRunner()


def test_vocab_indexing_is_clean(tmp_path, monkeypatch):
    """Test that vocab_terms collection only contains individual tokens, not full documents."""

    # create a corpus with a document that contains multiple tokens and newlines
    corpus_path = tmp_path / 'corpus.jsonl'
    corpus = [
        {'id': '1', 'text': 'Väinämöinen is wise.'},
        {'id': '2', 'text': 'Confidential \n'},
    ]
    with open(corpus_path, 'w', encoding='utf8') as f:
        for doc in corpus:
            f.write(json.dumps(doc) + '\n')

    work_dir = tmp_path / 'runs'
    run_name = 'test_vocab'

    # mock the embedding backend to avoid loading a real model and to be predictable
    class MockBackend:
        def __init__(self):
            self.dimension = 384
            self.model_name = 'mock-model'
            self.backend_type = 'sbert'

        def embed(self, texts):
            return np.zeros((len(texts), 384), dtype=np.float32)

    monkeypatch.setattr('pattern_nursery.cli.main.get_embedding_backend', lambda *args, **kwargs: MockBackend())

    # run init-index via CLI
    result = runner.invoke(app, [
        'init-index',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir),
        '--run-name', run_name,
        '--model-path', 'mock-model',
        '--collection', 'vocab_terms'
    ])

    assert result.exit_code == 0

    # load the index and check the stored texts
    manager = IndexManager(work_dir, run_name)
    store = manager.load_store('vocab_terms')

    # check if any 'Confidential \n' or full document text leaked in.
    query_vec = np.zeros((1, 384), dtype=np.float32)
    results = store.query(query_vec, top_k=100)

    stored_texts = [meta['text'] for idx, score, meta in results[0]]

    print(f"Stored texts: {stored_texts}")

    # verify that the noisy doc didn't leak as a single entry
    assert 'Confidential \n' not in stored_texts
    # vocabulary building defaults to lower=True
    assert 'confidential' in stored_texts

    # verify that full doc didn't leak
    assert 'Väinämöinen is wise.' not in stored_texts
    assert 'väinämöinen' in stored_texts
    assert 'is' in stored_texts
    assert 'wise' in stored_texts

    # verify that the tokens are stripped
    for text in stored_texts:
        assert text == text.strip()
        assert '\n' not in text

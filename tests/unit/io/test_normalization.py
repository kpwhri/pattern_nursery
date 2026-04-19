from unittest.mock import MagicMock

import numpy as np
import pytest

from pattern_nursery.index.manager import IndexManager
from pattern_nursery.nlp.regex_engine import RegexEngine


@pytest.fixture
def mock_backend():
    backend = MagicMock()
    backend.dimension = 384
    backend.model_name = 'mock_model'

    # return random vectors but same for same text for some consistency in expansion tests
    def embed_func(texts):
        if isinstance(texts, str):
            texts = [texts]
        return np.random.RandomState(42).rand(len(texts), 384).astype(np.float32)

    backend.embed.side_effect = embed_func
    return backend


def test_normalization_in_vocab_building(tmp_path, mock_backend):
    engine = RegexEngine()
    manager = IndexManager(tmp_path, 'test_norm', nlp_engine=engine)

    corpus = [
        ('1', 'Heart failure and heart attack.'),
        ('2', 'HEART diseases.')
    ]

    # build_and_persist_vocabulary with lower=True (default)
    vocab = manager.build_and_persist_vocabulary(corpus)

    # RegexEngine with \b\w+\b will find: Heart, failure, and, heart, attack, HEART, diseases
    # Lowercased: heart, failure, and, heart, attack, heart, diseases
    # Unique: and, attack, diseases, failure, heart

    assert 'heart' in vocab
    assert 'Heart' not in vocab
    assert 'HEART' not in vocab
    assert 'failure' in vocab

    vocab_file = tmp_path / 'runs' / 'test_norm' / 'artifacts' / 'vocab.txt'
    with open(vocab_file, 'r') as f:
        content = f.read().splitlines()

    assert 'heart' in content
    assert 'Heart' not in content


def test_normalization_in_expand_terms(tmp_path, mock_backend):
    from pattern_nursery.expansion.term_guided import expand_terms

    # Mock store
    store = MagicMock()
    # Mock query to return something
    store.query.return_value = [[(0, 0.9, {'text': 'heart'})]]

    seeds = ['HEART']
    expanded = expand_terms(seeds, store, mock_backend, lower=True)

    # Check that it called embed with 'heart'
    mock_backend.embed.assert_called_with(['heart'])
    assert expanded[0]['seed'] == 'heart'


def test_regex_engine_tokenize_lower():
    engine = RegexEngine()
    text = "Hello WORLD"
    tokens = engine.tokenize(text, lower=True)
    assert tokens == ["Hello", "world"] or tokens == ["hello", "world"]
    # tokens = self.tok_pattern.findall(text) -> ['Hello', 'WORLD']
    # if lower: [t.lower() for t in tokens] -> ['hello', 'world']
    assert "hello" in tokens
    assert "world" in tokens

    tokens_no_lower = engine.tokenize(text, lower=False)
    assert "Hello" in tokens_no_lower
    assert "WORLD" in tokens_no_lower

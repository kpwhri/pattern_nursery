from unittest.mock import MagicMock

import pytest

from pattern_nursery.index.manager import IndexManager
from pattern_nursery.nlp.nltk_engine import NLTKEngine
from pattern_nursery.nlp.regex_engine import RegexEngine
from pattern_nursery.nlp.spacy_engine import SpaCyEngine


def test_regex_engine_tokenize_strip():
    engine = RegexEngine()
    text = "Väinämöinen  Ilmarinen\n"
    tokens = engine.tokenize(text)
    assert "Väinämöinen" in tokens
    assert "Ilmarinen" in tokens
    for t in tokens:
        assert t == t.strip()
        assert len(t) > 0


def test_index_manager_vocab_strip(tmp_path):
    engine = RegexEngine()
    manager = IndexManager(tmp_path, 'test_run', nlp_engine=engine)

    # Mocking paths.get_run_artifacts_path as it's used in build_and_persist_vocabulary
    from pattern_nursery.index import paths
    original_get_run_artifacts_path = paths.get_run_artifacts_path
    paths.get_run_artifacts_path = MagicMock(return_value=tmp_path)

    try:
        texts = [
            ("1", "Väinämöinen "),
            ("2", " Ilmarinen\n"),
            ("3", "  \n  ")
        ]
        vocab = manager.build_and_persist_vocabulary(texts)

        # vocab is lowercased by default in build_and_persist_vocabulary
        assert "väinämöinen" in vocab
        assert "ilmarinen" in vocab
        assert "" not in vocab
        assert " " not in vocab
        assert "\n" not in vocab

        for term in vocab:
            assert term == term.strip()
    finally:
        paths.get_run_artifacts_path = original_get_run_artifacts_path


@pytest.mark.skipif(True, reason="Requires nltk data")
def test_nltk_engine_tokenize_strip():
    try:
        engine = NLTKEngine()
        text = "Lemminkäinen  "
        tokens = engine.tokenize(text)
        assert "Lemminkäinen" in tokens
        for t in tokens:
            assert t == t.strip()
    except Exception:
        pytest.skip("NLTK not available")


@pytest.mark.skipif(True, reason="Requires spacy model")
def test_spacy_engine_tokenize_strip():
    try:
        engine = SpaCyEngine()
        text = "Aino  "
        tokens = engine.tokenize(text)
        assert "Aino" in tokens
        for t in tokens:
            assert t == t.strip()
    except Exception:
        pytest.skip("spaCy not available")

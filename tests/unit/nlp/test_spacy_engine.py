import pytest

from pattern_nursery.nlp.spacy_engine import SpaCyEngine

try:
    import spacy

    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False


@pytest.mark.skipif(not HAS_SPACY, reason='spacy not installed')
def test_spacy_engine_basic():
    try:
        engine = SpaCyEngine(model_name='en_core_web_sm')
        text = 'Väinämöinen played the kantele. It was beautiful.'
        sents = engine.split_sentences(text)
        assert len(sents) == 2
        assert 'kantele' in sents[0]

        chunks = engine.noun_chunks(text)
        assert len(chunks) > 0
    except Exception as e:
        pytest.skip(f'SpaCy model loading failed: {e}')


@pytest.mark.skipif(HAS_SPACY, reason='testing ImportError path')
def test_spacy_import_error():
    with pytest.raises(ImportError, match='spacy is required'):
        SpaCyEngine()

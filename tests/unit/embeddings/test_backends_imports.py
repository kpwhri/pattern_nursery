import pytest

from pattern_nursery.embeddings.sentence_transformers_backend import SentenceTransformersBackend

# check if sentence_transformers is available
try:
    import sentence_transformers

    HAS_ST = True
except ImportError:
    HAS_ST = False


@pytest.mark.skipif(not HAS_ST, reason='sentence-transformers not installed')
def test_st_backend_dimension(tmp_path):
    """Test ST backend dimension (using a dummy path or local model if available)."""
    # TODO
    pass


@pytest.mark.skipif(HAS_ST, reason='testing ImportError path')
def test_st_backend_import_error():
    with pytest.raises(ImportError, match='sentence-transformers is required'):
        SentenceTransformersBackend('some/path')


try:
    import gensim

    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False

from pattern_nursery.embeddings.fasttext_backend import FastTextBackend
from pattern_nursery.embeddings.word2vec_backend import Word2VecBackend


@pytest.mark.skipif(HAS_GENSIM, reason='testing ImportError path')
def test_gensim_backends_import_error():
    with pytest.raises(ImportError, match='gensim is required'):
        FastTextBackend('some/path')
    with pytest.raises(ImportError, match='gensim is required'):
        Word2VecBackend('some/path')

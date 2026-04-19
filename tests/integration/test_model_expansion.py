from unittest.mock import MagicMock

from pattern_nursery.expansion import model_guided


def test_expand_terms_with_model():
    backend = MagicMock()
    backend.get_similar_terms.return_value = [
        {'candidate': 'ilmarinen', 'score': 0.95, 'source': 'model_guided'},
        {'candidate': 'joukahainen', 'score': 0.85, 'source': 'model_guided'}
    ]

    seeds = ['väinämöinen']
    results = model_guided.expand_terms_with_model(seeds, backend, top_k=2, threshold=0.9)

    # only Ilmarinen should be returned because of threshold 0.9
    assert len(results) == 1
    assert results[0]['seed'] == 'väinämöinen'
    assert results[0]['candidate'] == 'ilmarinen'
    assert results[0]['score'] == 0.95


def test_expand_terms_with_model_unsupported():
    # backend that doesn't implement it (returns empty list by default)
    backend = MagicMock()
    backend.get_similar_terms.return_value = []

    seeds = ['lemmitkäinen']
    results = model_guided.expand_terms_with_model(seeds, backend)

    assert len(results) == 0

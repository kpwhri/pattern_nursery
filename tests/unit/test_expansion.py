from unittest.mock import MagicMock

import numpy as np

from pattern_nursery.expansion import term_guided


def test_expand_terms():
    store = MagicMock()
    # mock query result: List[List[Tuple[int, float, Dict[str, Any]]]]
    store.query.return_value = [[(0, 0.9, {'text': 'expanded_term'})]]

    backend = MagicMock()
    backend.embed.return_value = np.random.rand(1, 384)

    results = term_guided.expand_terms(['seed'], store, backend, top_k=1, threshold=0.5)

    assert len(results) == 1
    assert results[0]['seed'] == 'seed'
    assert results[0]['candidate'] == 'expanded_term'
    assert results[0]['score'] == 0.9

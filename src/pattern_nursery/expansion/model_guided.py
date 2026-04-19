from typing import List, Dict, Any

from pattern_nursery.embeddings.base import EmbeddingBackend


def expand_terms_with_model(
        seeds: List[str],
        backend: EmbeddingBackend,
        top_k: int = 10,
        threshold: float = 0.0,
        lower: bool = True
) -> List[Dict[str, Any]]:
    """Expand seed terms using the embedding model's built-in vocabulary."""
    if lower:
        seeds = [s.lower() for s in seeds]

    expanded = []
    for seed in seeds:
        results = backend.get_similar_terms(seed, top_k=top_k)
        for res in results:
            if res['score'] >= threshold:
                expanded.append({
                    'seed': seed,
                    'candidate': res['candidate'],
                    'score': res['score'],
                    'source': 'model_guided'
                })
    return expanded

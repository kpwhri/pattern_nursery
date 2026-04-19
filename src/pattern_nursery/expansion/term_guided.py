from typing import List, Dict, Any


def expand_terms(
        seeds: List[str], store, backend, top_k: int = 10, threshold: float = 0.0, lower: bool = True
) -> List[Dict[str, Any]]:
    """Expand seed terms using vocabulary vector store."""
    if lower:
        seeds = [s.lower() for s in seeds]
    query_vectors = backend.embed(seeds)
    results = store.query(query_vectors, top_k=top_k)

    expanded = []
    for i, seed in enumerate(seeds):
        for idx, score, meta in results[i]:
            if score >= threshold:
                expanded.append({
                    'seed': seed,
                    'candidate': meta['text'],
                    'score': score,
                    'source': 'term_guided'
                })
    return expanded

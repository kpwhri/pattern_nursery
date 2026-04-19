from typing import List, Union

import numpy as np

from pattern_nursery.embeddings.base import EmbeddingBackend


class FastTextBackend(EmbeddingBackend):
    """FastText embedding backend using gensim."""

    def __init__(self, model_path: str):
        self._model_path = model_path
        try:
            from gensim.models import FastText
            self.model = FastText.load(model_path)
        except ImportError:
            raise ImportError('gensim is required for FastTextBackend')

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            # simple average of word vectors
            words = text.split()
            if not words:
                embeddings.append(np.zeros(self.dimension))
                continue

            vecs = [self.model.wv[w] for w in words if w in self.model.wv]
            if not vecs:
                embeddings.append(np.zeros(self.dimension))
            else:
                embeddings.append(np.mean(vecs, axis=0))

        return np.array(embeddings)

    @property
    def dimension(self) -> int:
        return self.model.vector_size

    @property
    def model_name(self) -> str:
        return self._model_path

    def get_similar_terms(self, term: str, top_k: int = 10) -> List[dict]:
        """Get similar terms directly from the FastText model."""
        # fasttext can construct vectors for OOV words via subwords
        try:
            results = self.model.wv.most_similar(term, topn=top_k)
            return [
                {'candidate': word, 'score': float(score), 'source': 'model_guided'}
                for word, score in results
            ]
        except KeyError:
            # if even after n-gram it fails
            return []

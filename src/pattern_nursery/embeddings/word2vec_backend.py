from typing import List, Union

import numpy as np

from pattern_nursery.embeddings.base import EmbeddingBackend


class Word2VecBackend(EmbeddingBackend):
    """Word2Vec embedding backend using gensim."""

    def __init__(self, model_path: str):
        self._model_path = model_path
        try:
            from gensim.models import KeyedVectors
            # load from binary or text
            if model_path.endswith('.bin'):
                self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
            else:
                self.model = KeyedVectors.load_word2vec_format(model_path, binary=False)
        except ImportError:
            raise ImportError('gensim is required for Word2VecBackend')

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            words = text.split()
            if not words:
                embeddings.append(np.zeros(self.dimension))
                continue

            vecs = [self.model[w] for w in words if w in self.model]
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
        """Get similar terms directly from the Word2Vec model."""
        if term not in self.model:
            return []

        results = self.model.most_similar(term, topn=top_k)
        return [
            {'candidate': word, 'score': float(score), 'source': 'model_guided'}
            for word, score in results
        ]

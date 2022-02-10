from typing import Dict, Tuple

from gensim.models import KeyedVectors

SimilarityKey = Tuple[str, str]


class SimilarityCache:
    def __init__(self, model: KeyedVectors):
        self._cache: Dict[SimilarityKey, float] = {}
        self._model = model

    def __getitem__(self, key: SimilarityKey) -> float:
        return self.similarity(*key)

    def __setitem__(self, key: SimilarityKey, value: float):
        self._cache[(key[0], key[1])] = value
        self._cache[(key[1], key[0])] = value

    def similarity(self, word1: str, word2: str) -> float:
        try:
            return self._cache[(word1, word2)]
        except KeyError:
            similarity = self._model.similarity(word1, word2)
            self[(word1, word2)] = similarity
            return similarity

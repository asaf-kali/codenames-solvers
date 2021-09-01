from functools import lru_cache

from gensim.models import KeyedVectors

DATA_FOLDER = "language_data"


@lru_cache
def load_language(language: str) -> KeyedVectors:
    return KeyedVectors.load_word2vec_format(f"{DATA_FOLDER}/{language}.bin", binary=True)

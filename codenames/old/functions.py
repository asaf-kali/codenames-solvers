from typing import List, Union

import numpy as np
from words import WORD_VECTORS, WORDS

FORBIDDEN_WORDS = {"ה", "</s>"}


def find_word_indices(word: str) -> List[int]:
    indices = []
    for i, other in enumerate(WORDS):
        if word in other:
            indices.append(i)
    return indices


def find_word_index(word: str) -> int:
    return find_word_indices(word)[0]


def n_closest_words(word: Union[str, np.ndarray], n: int = 10) -> List[str]:
    if isinstance(word, str):
        word_index = find_word_index(word)
        word_vector = WORD_VECTORS[word_index, :]
    else:
        word_vector = word
    close_word_vectors: np.ndarray = WORD_VECTORS @ word_vector.T
    close_word_vectors = close_word_vectors.argsort()
    indices = reversed(close_word_vectors[-n:])
    close_words = [WORDS[i] for i in indices if WORDS[i] not in FORBIDDEN_WORDS]
    return close_words


def find_vector(word: str) -> np.ndarray:
    word_index = find_word_index(word)
    return WORD_VECTORS[word_index]

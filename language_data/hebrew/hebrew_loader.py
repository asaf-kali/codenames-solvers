import logging
import os
from typing import List

import numpy as np
from gensim.models import KeyedVectors

log = logging.getLogger(__name__)

VECTOR_LENGTH = 100


def load_hebrew_model(language_base_folder: str, model_name: str) -> KeyedVectors:
    log.debug(f"Loading Hebrew model: {model_name}...")
    model = KeyedVectors(vector_size=VECTOR_LENGTH)
    log.debug("Loading raw data...")
    vectors = _load_vectors(language_base_folder, model_name)
    words = _load_words(language_base_folder, model_name)
    word_number = len(words)
    vectors = vectors[:-10].reshape(word_number, VECTOR_LENGTH)
    log.debug("Normalizing...")
    vectors = vectors / np.linalg.norm(vectors, axis=-1)[:, np.newaxis]
    log.debug("Adding to model...")
    model.add_vectors(keys=words, weights=vectors)
    log.debug("Load Hebrew model done")
    return model


def _load_words(language_base_folder: str, model_name: str) -> List[str]:
    words_file_name = os.path.join(language_base_folder, model_name, "words.txt")
    with open(words_file_name, encoding="utf-8") as words_file:
        words = [word.replace("\n", "") for word in words_file.readlines()]
    return words


def _load_vectors(language_base_folder: str, model_name: str) -> np.ndarray:
    vectors_file_name = os.path.join(language_base_folder, model_name, "vectors.npy")
    vectors = np.fromfile(vectors_file_name)
    return vectors

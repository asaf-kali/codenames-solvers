import logging
from functools import lru_cache

from gensim.models import KeyedVectors

log = logging.getLogger(__name__)
DATA_FOLDER = "language_data"


@lru_cache
def load_language(language: str) -> KeyedVectors:
    log.debug(f"Loading language: {language}...")
    data = KeyedVectors.load_word2vec_format(f"{DATA_FOLDER}/{language}.bin", binary=True)
    log.debug("Language loaded")
    return data

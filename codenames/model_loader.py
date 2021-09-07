import logging
from functools import lru_cache

from gensim.models import KeyedVectors

log = logging.getLogger(__name__)
DATA_FOLDER = "language_data"


@lru_cache
def load_language(language: str, cleaned_model=False) -> KeyedVectors:
    log.debug(f"Loading language: {language}...")
    if cleaned_model:
        data_path = f"{DATA_FOLDER}/{language}_cleaned.bin"
    else:
        data_path = f"{DATA_FOLDER}/{language}.bin"
    data = KeyedVectors.load_word2vec_format(data_path, binary=True)
    log.debug("Language loaded")
    return data

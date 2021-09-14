import logging
import os
from functools import lru_cache

from gensim.models import KeyedVectors

from language_data.hebrew.hebrew_loader import load_hebrew_model

log = logging.getLogger(__name__)
MODEL_NAME_ENV_KEY = "SNA_MODEL_NAME"
LANGUAGE_DATA_FOLDER = "language_data"
DEFAULT_MODEL_NAME = "wiki-300"


@lru_cache
def load_language(language: str, model_name: str = None) -> KeyedVectors:
    if model_name is None:
        model_name = os.environ.get(key=MODEL_NAME_ENV_KEY, default=DEFAULT_MODEL_NAME)
    log.debug(f"Loading language: {language} (model: {model_name})...")
    language_base_folder = os.path.join(LANGUAGE_DATA_FOLDER, language)
    if language == "hebrew":
        return load_hebrew_model(language_base_folder, model_name)
    file_path = os.path.join(language_base_folder, f"{model_name}.bin")
    data = KeyedVectors.load_word2vec_format(file_path, binary=True)
    log.debug("Language loaded")
    return data

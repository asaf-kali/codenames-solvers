import logging
import os
from threading import Thread
from typing import Optional

from gensim.models import KeyedVectors

from codenames_solvers.models.cache import ModelCache
from codenames_solvers.models.identifier import ModelIdentifier

log = logging.getLogger(__name__)

MODEL_NAME_ENV_KEY = "MODEL_NAME"
IS_STEMMED_ENV_KEY = "IS_STEMMED"
DEFAULT_MODEL_NAME = "wiki-50"

model_cache = ModelCache()


def set_language_data_folder(language_data_folder: str):
    model_cache.language_data_folder = language_data_folder


def is_loaded(model_identifier: ModelIdentifier) -> bool:
    return model_cache.is_loaded(model_identifier)


def load_model(model_identifier: ModelIdentifier) -> KeyedVectors:
    return model_cache.load_model(model_identifier)


def load_model_async(model_identifier: ModelIdentifier):
    thread = Thread(target=load_model, args=(model_identifier,))
    thread.start()


def load_language(language: str, model_name: Optional[str] = None, is_stemmed: Optional[bool] = None) -> KeyedVectors:
    if model_name is None:
        model_name = os.environ.get(key=MODEL_NAME_ENV_KEY, default=DEFAULT_MODEL_NAME)  # type: ignore
    if is_stemmed is None:
        is_stemmed = bool(os.environ.get(key=IS_STEMMED_ENV_KEY, default=False))  # type: ignore
    model_identifier = ModelIdentifier(language=language, model_name=model_name, is_stemmed=is_stemmed)
    return load_model(model_identifier)


def load_language_async(language: str, model_name: Optional[str] = None, is_stemmed: Optional[bool] = None):
    thread = Thread(target=load_language, args=(language, model_name, is_stemmed), daemon=True)
    thread.start()


# def load_word2vec_format(language_base_folder: str, model_name: str) -> KeyedVectors:
#     # Where data structure is `*.bin`
#     file_path = os.path.join(language_base_folder, f"{model_name}.bin")
#     data = KeyedVectors.load_word2vec_format(file_path, binary=True)
#     return data

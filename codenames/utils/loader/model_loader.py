import logging
import os
from os.path import expanduser
from threading import Lock, Thread
from typing import Dict

from generic_iterative_stemmer.models import StemmedKeyedVectors
from gensim.models import KeyedVectors
from pydantic import BaseModel

log = logging.getLogger(__name__)

MODEL_NAME_ENV_KEY = "MODEL_NAME"
IS_STEMMED_ENV_KEY = "IS_STEMMED"
DEFAULT_MODEL_NAME = "wiki-50"


class ModelIdentifier(BaseModel):
    language: str
    model_name: str
    is_stemmed: bool = False

    def __hash__(self):
        return hash(f"{self.language}-{self.model_name}-{self.is_stemmed}")


class ModelCache:
    def __init__(self):
        self.language_data_folder = "~/.cache/language_data"
        self._cache: Dict[ModelIdentifier, KeyedVectors] = {}
        self._main_lock = Lock()
        self._model_locks: Dict[ModelIdentifier, Lock] = {}

    def _get_model_lock(self, model_identifier: ModelIdentifier) -> Lock:
        with self._main_lock:
            model_lock = self._model_locks.setdefault(model_identifier, Lock())
        return model_lock

    def load_model(self, model_identifier: ModelIdentifier) -> KeyedVectors:
        log.info(f"Loading model {model_identifier}")
        model_lock = self._get_model_lock(model_identifier)
        with model_lock:
            if model_identifier not in self._cache:
                self._cache[model_identifier] = self._load_model(model_identifier)
            return self._cache[model_identifier]

    def _load_model(self, model_identifier: ModelIdentifier) -> KeyedVectors:
        # TODO: in case loading fails, try gensim downloader
        # import gensim.downloader as api
        # model = api.load("wiki-he")
        log.debug("Loading model...", extra={"model": model_identifier})
        language_base_folder = expanduser(os.path.join(self.language_data_folder, model_identifier.language))
        model = load_kv_format(
            language_base_folder=language_base_folder,
            model_name=model_identifier.model_name,
            is_stemmed=model_identifier.is_stemmed,
        )
        log.debug("Model loaded", extra={"model": model_identifier})
        return model


_model_cache = ModelCache()


def set_language_data_folder(language_data_folder: str):
    _model_cache.language_data_folder = language_data_folder


def load_model(model_identifier: ModelIdentifier) -> KeyedVectors:
    return _model_cache.load_model(model_identifier)


def load_model_async(model_identifier: ModelIdentifier):
    t = Thread(target=load_model, args=(model_identifier,))
    t.start()


def load_language(language: str, model_name: str = None, is_stemmed: bool = None) -> KeyedVectors:
    if model_name is None:
        model_name = os.environ.get(key=MODEL_NAME_ENV_KEY, default=DEFAULT_MODEL_NAME)  # type: ignore
    if is_stemmed is None:
        is_stemmed = bool(os.environ.get(key=IS_STEMMED_ENV_KEY, default=False))  # type: ignore
    model_identifier = ModelIdentifier(language=language, model_name=model_name, is_stemmed=is_stemmed)
    return load_model(model_identifier)


def load_language_async(language: str, model_name: str = None, is_stemmed: bool = None):
    t = Thread(target=load_language, args=(language, model_name, is_stemmed), daemon=True)
    t.start()


# def load_word2vec_format(language_base_folder: str, model_name: str) -> KeyedVectors:
#     # Where data structure is `*.bin`
#     file_path = os.path.join(language_base_folder, f"{model_name}.bin")
#     data = KeyedVectors.load_word2vec_format(file_path, binary=True)
#     return data


def load_kv_format(language_base_folder: str, model_name: str, is_stemmed: bool = False) -> KeyedVectors:
    model_folder = os.path.join(language_base_folder, model_name)
    file_path = os.path.join(model_folder, "model.kv")  # TODO: This needs fixing
    if is_stemmed:
        model = StemmedKeyedVectors.load(file_path)
    else:
        model = KeyedVectors.load(file_path)
    return model

import logging
import os
from threading import Lock, Thread
from typing import Dict, Literal, NamedTuple

from gensim.models import KeyedVectors

log = logging.getLogger(__name__)

MODEL_NAME_ENV_KEY = "SNA_MODEL_NAME"
LANGUAGE_DATA_FOLDER = "language_data"
DEFAULT_MODEL_NAME = "wiki-50"
SupportedLanguage = Literal["english", "hebrew"]


class ModelIdentifier(NamedTuple):
    language: SupportedLanguage
    model_name: str


def _load_model(model_identifier: ModelIdentifier) -> KeyedVectors:
    # TODO: in case loading fails, try gensim downloader
    # import gensim.downloader as api
    #
    # model = api.load("wiki-he")

    log.debug("Loading language...", extra={"model": model_identifier})
    language_base_folder = os.path.join(LANGUAGE_DATA_FOLDER, model_identifier.language)
    model = load_kv_format(language_base_folder, model_identifier.model_name)
    log.debug("Language loaded", extra={"model": model_identifier})
    return model


class LanguageCache:
    def __init__(self):
        self._cache: Dict[ModelIdentifier, KeyedVectors] = {}
        self._main_lock = Lock()
        self._model_locks: Dict[ModelIdentifier, Lock] = {}

    def _get_model_lock(self, model_identifier: ModelIdentifier) -> Lock:
        with self._main_lock:
            model_lock = self._model_locks.setdefault(model_identifier, Lock())
        return model_lock

    def _get_model(self, model_identifier: ModelIdentifier) -> KeyedVectors:
        model_lock = self._get_model_lock(model_identifier)
        with model_lock:
            if model_identifier not in self._cache:
                self._cache[model_identifier] = _load_model(model_identifier)
            return self._cache[model_identifier]

    def load_language(self, language: SupportedLanguage, model_name: str = None) -> KeyedVectors:
        if model_name is None:
            model_name = os.environ.get(key=MODEL_NAME_ENV_KEY, default=DEFAULT_MODEL_NAME)
        model_identifier = ModelIdentifier(language, model_name)  # type: ignore
        return self._get_model(model_identifier)


_language_cache = LanguageCache()


def load_language(language: SupportedLanguage, model_name: str = None) -> KeyedVectors:
    return _language_cache.load_language(language, model_name)


def load_language_async(language: SupportedLanguage, model_name: str = None):
    t = Thread(target=load_language, args=(language, model_name))
    t.start()


def load_word2vec_format(language_base_folder: str, model_name: str) -> KeyedVectors:
    file_path = os.path.join(language_base_folder, f"{model_name}.bin")
    data = KeyedVectors.load_word2vec_format(file_path, binary=True)
    return data


def load_kv_format(
    language_base_folder: str,
    model_name: str,
) -> KeyedVectors:
    model_folder = os.path.join(language_base_folder, model_name)
    file_path = os.path.join(model_folder, f"{model_name}.kv")
    model: KeyedVectors = KeyedVectors.load(file_path)  # type: ignore
    return model

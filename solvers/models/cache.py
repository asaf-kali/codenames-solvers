import logging
import os
from os.path import expanduser
from threading import Lock
from typing import Dict

from generic_iterative_stemmer.models import StemmedKeyedVectors
from gensim.models import KeyedVectors

from solvers.models.identifier import ModelIdentifier

log = logging.getLogger(__name__)


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

    def is_loaded(self, model_identifier: ModelIdentifier) -> bool:
        return model_identifier in self._cache

    def load_model(self, model_identifier: ModelIdentifier) -> KeyedVectors:
        model_lock = self._get_model_lock(model_identifier)
        with model_lock:
            if not self.is_loaded(model_identifier):
                self._cache[model_identifier] = self._load_model(model_identifier)
            return self._cache[model_identifier]

    def _load_model(self, model_identifier: ModelIdentifier) -> KeyedVectors:
        # TODO: in case loading fails, try gensim downloader
        # import gensim.downloader as api
        # model = api.load("wiki-he")
        log.info("Loading model...", extra={"model": model_identifier.dict()})
        language_base_folder = expanduser(os.path.join(self.language_data_folder, model_identifier.language))
        model = load_kv_format(
            language_base_folder=language_base_folder,
            model_name=model_identifier.model_name,
            is_stemmed=model_identifier.is_stemmed,
        )
        log.info("Model loaded", extra={"model": model_identifier.dict()})
        return model


def load_kv_format(language_base_folder: str, model_name: str, is_stemmed: bool = False) -> KeyedVectors:
    model_folder = os.path.join(language_base_folder, model_name)
    file_path = os.path.join(model_folder, "model.kv")  # TODO: This needs fixing
    if is_stemmed:
        model = StemmedKeyedVectors.load(file_path)
    else:
        model = KeyedVectors.load(file_path)
    return model

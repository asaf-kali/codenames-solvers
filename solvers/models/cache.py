import logging
import os
from os.path import expanduser
from threading import Lock
from typing import Dict

import gensim.downloader as gensim_api
from generic_iterative_stemmer.models import StemmedKeyedVectors
from gensim.models import KeyedVectors

from solvers.models.identifier import ModelIdentifier

log = logging.getLogger(__name__)


class ModelCache:
    def __init__(self, language_data_folder: str = "~/.cache/language_data"):
        self.language_data_folder = language_data_folder
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
        log.info("Loading model...", extra={"model": model_identifier.dict()})
        language_base_folder = expanduser(os.path.join(self.language_data_folder, model_identifier.language))
        try:
            return load_kv_format(
                language_base_folder=language_base_folder,
                model_name=model_identifier.model_name,
                is_stemmed=model_identifier.is_stemmed,
            )
        except Exception as e:
            log.warning(f"Failed to load model: {e}", exc_info=True)
            return load_from_gensim(model_identifier)


def load_kv_format(language_base_folder: str, model_name: str, is_stemmed: bool = False) -> KeyedVectors:
    model_folder = os.path.join(language_base_folder, model_name)
    file_path = os.path.join(model_folder, "model.kv")
    log.debug(f"Looking for [{model_name}] in {file_path}...")
    if is_stemmed:
        model = StemmedKeyedVectors.load(file_path)
    else:
        model = KeyedVectors.load(file_path)
    log.debug(f"Successfully loaded [{model_name}] from {file_path}")
    return model


def load_from_gensim(model_identifier: ModelIdentifier) -> KeyedVectors:
    log.debug(f"Looking for [{model_identifier.model_name}] in gensim API...")
    model = gensim_api.load(model_identifier.model_name)
    log.debug(f"Successfully loaded [{model_identifier.model_name}] from gensim API")
    return model

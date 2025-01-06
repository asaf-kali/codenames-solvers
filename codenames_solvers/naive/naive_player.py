from typing import Optional

from codenames.generic.board import Board
from codenames.generic.card import CardColor
from codenames.generic.player import Player
from codenames.generic.team import Team
from gensim.models import KeyedVectors

from codenames_solvers.models import (
    DefaultFormatAdapter,
    ModelFormatAdapter,
    ModelIdentifier,
    load_language,
    load_model,
)


class NaivePlayer[C: CardColor, T: Team](Player[C, T]):
    def __init__(
        self,
        name: str,
        team: T,
        model: Optional[KeyedVectors] = None,
        model_identifier: Optional[ModelIdentifier] = None,
        model_adapter: Optional[ModelFormatAdapter] = None,
    ):
        super().__init__(name=name, team=team)
        self._model = model
        self._model_identifier = model_identifier
        self._model_adapter = model_adapter or DefaultFormatAdapter()

    @property
    def model(self) -> KeyedVectors:
        if not self._model:
            self._prepare_model()
        return self._model

    def on_game_start(self, board: Board):
        self._prepare_model(language=board.language)

    def model_format(self, word: str) -> str:
        return self._model_adapter.to_model_format(word)

    def board_format(self, word: str) -> str:
        return self._model_adapter.to_board_format(word)

    def _prepare_model(self, language: Optional[str] = None):
        if self._model_identifier and self._model_identifier.language == language:
            self._model = load_model(model_identifier=self._model_identifier)
        elif language:
            self._model = load_language(language=language)
        else:
            raise ValueError("No language provided")
        self._model_adapter.clear_cache()
        self._model_adapter.checker = self.model.__contains__

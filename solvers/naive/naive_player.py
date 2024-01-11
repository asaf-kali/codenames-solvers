from abc import ABC
from typing import Optional

from codenames.game.board import Board
from codenames.game.color import TeamColor
from codenames.game.player import Player
from gensim.models import KeyedVectors

from solvers.models import (
    DefaultFormatAdapter,
    ModelFormatAdapter,
    ModelIdentifier,
    load_language,
    load_model,
)


class NaivePlayer(Player, ABC):
    def __init__(
        self,
        name: str,
        team_color: TeamColor,
        model: Optional[KeyedVectors] = None,
        model_identifier: Optional[ModelIdentifier] = None,
        model_adapter: Optional[ModelFormatAdapter] = None,
    ):
        super().__init__(name=name, team_color=team_color)
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
        self._model_adapter.checker = self.model.key_to_index.__contains__

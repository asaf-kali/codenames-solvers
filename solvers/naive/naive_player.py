from abc import ABC
from typing import Optional

from codenames.game.board import Board
from codenames.game.player import Player
from gensim.models import KeyedVectors

from solvers.models import (
    DEFAULT_MODEL_ADAPTER,
    ModelFormatAdapter,
    ModelIdentifier,
    load_language,
    load_model,
)


class NaivePlayer(Player, ABC):
    def __init__(
        self,
        name: str,
        model: Optional[KeyedVectors] = None,
        model_identifier: Optional[ModelIdentifier] = None,
        model_adapter: Optional[ModelFormatAdapter] = None,
    ):
        super().__init__(name=name)
        self.model: KeyedVectors = model
        self.model_identifier = model_identifier
        self.model_adapter = model_adapter or DEFAULT_MODEL_ADAPTER

    def on_game_start(self, board: Board):
        if self.model_identifier and self.model_identifier.language == board.language:
            self.model = load_model(model_identifier=self.model_identifier)
        else:
            self.model = load_language(language=board.language)

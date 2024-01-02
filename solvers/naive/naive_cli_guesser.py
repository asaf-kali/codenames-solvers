import logging

import matplotlib.pyplot as plt
import numpy as np
from codenames.game.board import Board
from codenames.game.move import Guess
from codenames.game.state import GuesserGameState
from gensim.models import KeyedVectors

from solvers.cli import CliGuesser
from solvers.models import DEFAULT_MODEL_ADAPTER, ModelFormatAdapter, load_language

log = logging.getLogger(__name__)


class ModelAwareCliGuesser(CliGuesser):
    def __init__(
        self, name: str, model: KeyedVectors = None, model_adapter: ModelFormatAdapter = DEFAULT_MODEL_ADAPTER
    ):
        super().__init__(name=name)
        self.model: KeyedVectors = model
        self.model_adapter = model_adapter

    def on_game_start(self, board: Board):
        self.model = load_language(language=board.language)

    def guess(self, game_state: GuesserGameState) -> Guess:
        self.show_sims(game_state)
        return super().guess(game_state)

    def show_sims(self, game_state: GuesserGameState):
        current_hint_word = game_state.current_hint.formatted_word
        model_formatted_hint = self.model_format(current_hint_word)
        sims = []
        annotations = []
        for card in game_state.board.cards:
            word = str(card)
            if card.revealed:
                sim = np.nan
                annotation = f"{word[::-1]}:-"
            else:
                sim = self.model.similarity(model_formatted_hint, word)
                annotation = f"{word[::-1]}:{sim:.2f}"
            sims.append(sim)
            annotations.append(annotation)
        # plot
        fig, ax = plt.subplots()  # pylint disable=invalid-name
        sims_to_show = np.zeros((5, 5))
        # Loop over data dimensions and create text annotations.
        for i, (ann, sim) in enumerate(zip(annotations, sims, strict=True)):  # type: ignore
            row = i // 5
            col = i % 5
            sims_to_show[row, col] = sim
            _ = ax.text(col, row, ann, ha="center", va="center", color="w")
        ax.imshow(sims_to_show)
        title = f"{model_formatted_hint[::-1]}:{game_state.current_hint.card_amount}"
        ax.set_title(title)
        plt.grid(False)
        fig.set_size_inches(10, 10)
        plt.show(block=True)

    def model_format(self, word: str) -> str:
        return self.model_adapter.to_model_format(word)

    def board_format(self, word: str) -> str:
        return self.model_adapter.to_board_format(word)

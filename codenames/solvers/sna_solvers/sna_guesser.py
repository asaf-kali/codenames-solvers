from codenames import model_loader
from codenames.game.base import GuesserGameState, Guess
from codenames.game.player import Guesser
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from codenames.game.base import TeamColor, Hint, Board, HinterGameState
from typing import Dict, List, Tuple, Optional, Iterable, Sequence, Union

from codenames.model_loader import load_language
from codenames.solvers.sna_solvers.sna_hinter import format_word


class SnaGuesser(Guesser):

    def __init__(self, name: str, team_color: TeamColor):
        super().__init__(name=name, team_color=team_color)
        self.model: Optional[KeyedVectors] = None
        # self.language_length: Optional[int] = None
        # self.board_data: Optional[pd.DataFrame] = None


    def notify_game_starts(self, language: str, board: Board):
        self.model = load_language(language=language)



    def guess(self, state: GuesserGameState) -> Guess:
        optional_words = [card.word for card in state.board.unrevealed_cards]
        guess_word = self.model.most_similar_to_given(state.current_hint.word, optional_words)
        guess_idx = state.board.all_words.index(guess_word)
        guess = Guess(card_index=guess_idx)
        return guess

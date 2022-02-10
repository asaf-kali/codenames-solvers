import copy
from typing import List

import pandas as pd

from codenames.game import CardColor
from codenames.solvers.olympic.similarity_cache import SimilarityCache


class BoardHeuristic:
    def __init__(self, alpha: float, delta: float, similarity_cache: SimilarityCache, board_words: List[str]):
        self.alpha = alpha
        self.delta = delta
        self.board_words = board_words
        self.similarity_cache = similarity_cache
        board_size = len(board_words)
        self.memory = pd.DataFrame(
            {
                CardColor.BLUE: [1] * board_size,
                CardColor.RED: [1] * board_size,
                CardColor.GRAY: [1] * board_size,
                CardColor.BLACK: [1] * board_size,
            }
        )
        self.normalize_memory()

    def __deepcopy__(self, memodict=None):
        new = BoardHeuristic(self.alpha, self.delta, self.similarity_cache, self.board_words)
        new.memory = self.memory.copy()
        return new

    def normalize_memory(self):
        self.memory = self.memory.div(self.memory.sum(axis=1), axis=0)

    def get_updated_board_heuristic(self, hint: str, team_color: CardColor) -> "BoardHeuristic":
        opponent_color = team_color.opponent
        copy_memory = copy.deepcopy(self)
        for i, word in enumerate(self.board_words):
            similarity = max(self.similarity_cache.similarity(hint, word), 0)
            copy_memory.memory[team_color][i] += self.alpha * similarity + self.delta
            copy_memory.memory[opponent_color][i] += self.delta
            copy_memory.memory[CardColor.GRAY][i] += self.delta
            copy_memory.memory[CardColor.BLACK][i] += self.delta
        copy_memory.normalize_memory()
        return copy_memory

    def get_scores_for_color(self, color: CardColor) -> pd.Series:
        return self.memory[color].values

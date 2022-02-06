import copy

import pandas as pd
from gensim.models import KeyedVectors

from codenames.game import Board, CardColor


class Memory:
    def __init__(self, alpha: float, delta: float, model: KeyedVectors, board: Board):
        self.alpha = alpha
        self.delta = delta
        self.model = model
        self.board = board
        self.memory = pd.DataFrame(
            {
                CardColor.BLUE: [1] * self.board.size,
                CardColor.RED: [1] * self.board.size,
                CardColor.GRAY: [1] * self.board.size,
                CardColor.BLACK: [1] * self.board.size,
            }
        )
        self.normalize_memory()

    def __deepcopy__(self, memodict=None):
        new = Memory(self.alpha, self.delta, self.model, self.board)
        new.memory = self.memory.copy()
        return new

    def normalize_memory(self):
        self.memory = self.memory.div(self.memory.sum(axis=1), axis=0)

    def get_updated_memory(self, word: str, team_color: CardColor) -> "Memory":
        opponent_color = team_color.opponent
        copy_memory = copy.deepcopy(self)
        for i in range(self.board.size):
            card = self.board[i]
            similarity = max(self.model.similarity(word, card.word), 0)
            copy_memory.memory[team_color][i] += self.alpha * similarity + self.delta
            copy_memory.memory[opponent_color][i] += self.delta
            copy_memory.memory[CardColor.GRAY][i] += self.delta
            copy_memory.memory[CardColor.BLACK][i] += self.delta
        copy_memory.normalize_memory()
        return copy_memory

    def get_scores_for_color(self, color: CardColor) -> pd.Series:
        return self.memory[color].values

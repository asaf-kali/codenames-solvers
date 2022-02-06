import copy

import pandas as pd
from gensim.models import KeyedVectors

from codenames.game import Board, CardColor, TeamColor


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
        for color in self.memory.columns:
            column = self.memory[color]
            self.memory[color] = column / column.sum()

    def update_memory(self, word: str, team_color: TeamColor) -> "Memory":
        turn_color = team_color.as_card_color
        opponent_color = turn_color.opponent
        copy_memory = copy.deepcopy(self)
        for i in range(self.board.size):
            card = self.board[i]
            similarity = max(self.model.similarity(word, card.word), 0)
            copy_memory.memory[turn_color][i] += self.alpha * similarity + self.delta
            copy_memory.memory[opponent_color][i] += self.delta
            copy_memory.memory[CardColor.GRAY][i] += self.delta
            copy_memory.memory[CardColor.BLACK][i] += self.delta
        copy_memory.normalize_memory()
        return copy_memory

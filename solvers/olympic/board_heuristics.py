from typing import List, NamedTuple

import numpy as np
from codenames.game import CardColor
from gensim.models import KeyedVectors

SimilaritiesMatrix = np.ndarray
HeuristicsTensor = np.ndarray


def get_card_color_index(card_color: CardColor) -> int:
    return {
        CardColor.BLUE: 0,
        CardColor.RED: 1,
        CardColor.GRAY: 2,
        CardColor.BLACK: 3,
    }[card_color]


def normalize_vectors(u: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(u, axis=1)
    normalized = np.divide(u.T, norms).T
    return normalized


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> SimilaritiesMatrix:
    """
    Compute cosine similarity between two word matrices.
    Each row in u and v is a word vector (the first dimension).
    :return: A cosine_similarities matrix, where cosine_similarities[i, j] = cosine_similarity(u[i], v[j])
    """
    u_normalized = normalize_vectors(u)
    v_normalized = normalize_vectors(v)
    return (u_normalized @ v_normalized.T).T


class HeuristicsResult(NamedTuple):
    similarities: SimilaritiesMatrix
    heuristics: HeuristicsTensor


class HeuristicsCalculator:
    def __init__(
        self,
        model: KeyedVectors,
        board_words: List[str],
        current_heuristic: HeuristicsTensor,
        team_card_color: CardColor,
        alpha: float,
        delta: float,
    ):
        self.model = model
        self.board_words = board_words
        self.current_heuristic = current_heuristic
        self.team_card_color = team_card_color
        self.alpha = alpha
        self.delta = delta

    def calculate_similarities_to_board(self, vocabulary: List[str]) -> SimilaritiesMatrix:
        """
        Calculate similarities between words in vocabulary and words in the board.
        :return: a Similarities matrix `similarities`, where
        similarities[i, j] = cosine_similarity(vocabulary[i], board[j])
        """
        vocabulary_vectors = self.model[vocabulary]
        board_vectors = np.array([self.model[word] for word in self.board_words])
        cosine_similarities = cosine_similarity(board_vectors, vocabulary_vectors)
        return cosine_similarities

    def calculate_heuristics_for_vocabulary(self, vocabulary: List[str]) -> HeuristicsResult:
        similarities = self.calculate_similarities_to_board(vocabulary=vocabulary)
        heuristics = self.calculate_heuristics_for_similarities(similarities=similarities)
        return HeuristicsResult(similarities=similarities, heuristics=heuristics)

    def calculate_heuristics_for_similarities(self, similarities: SimilaritiesMatrix) -> HeuristicsTensor:
        """
        Calculate updated board heuristic for each word in the vocabulary.
        :return: a Probability tensor `heuristics`, where
        heuristics[i, j, k] = P(card[j].color = colors[k] | given_hint = vocabulary[i])
        """
        vocabulary_size = similarities.shape[0]
        my_color_index = get_card_color_index(self.team_card_color)
        # Futures shape: (vocabulary_size, 25_board_size, 4_card_colors)
        futures = np.array([self.current_heuristic] * vocabulary_size)

        alpha = np.zeros(shape=futures.shape)
        alpha[:, :, my_color_index] = self.alpha * np.maximum(similarities, 0)
        # TODO: Alpha can be negative, think about how to treat this.

        result = futures + self.delta + alpha
        normalized_result = result / result.sum(axis=2)[:, :, None]
        return normalized_result

import logging
from functools import cached_property
from typing import Iterable, List, NamedTuple, Optional, Tuple
from uuid import uuid4

import numpy as np
from gensim.models import KeyedVectors

from codenames.game import DEFAULT_MODEL_ADAPTER, Hinter, ModelFormatAdapter
from codenames.game.base import (
    Board,
    CardColor,
    GivenHint,
    Hint,
    HinterGameState,
    WordGroup,
)
from codenames.utils.loader.model_loader import load_language

log = logging.getLogger(__name__)


class NoProposalsFound(Exception):
    pass


def unrevealed_cards_mask(board: Board) -> np.ndarray:
    return np.array([not card.revealed for card in board])


def card_color_mask(board: Board, card_color: CardColor) -> np.ndarray:
    idxs = [i for i, card in enumerate(board) if card.color == card_color]
    mask = np.zeros(board.size, dtype=bool)
    mask[idxs] = True
    return mask


class OlympicProposal:
    def __init__(
        self,
        hint_word: str,
        best_nth_words: WordGroup,
        best_nth_similarities: Tuple[float, ...],
        worst: str,
        worst_similarity: float,
        olympic_ratio: float,
        **kwargs,
    ):
        self.hint_word = hint_word
        self.best_nth_words = best_nth_words
        self.best_nth_similarities = best_nth_similarities
        self.worst = worst
        self.worst_similarity = worst_similarity
        self.olympic_ratio = olympic_ratio
        self.extra = kwargs
        self.grade = self.calculate_grade()

    def __str__(self) -> str:
        return f"hint={self.hint_word} for={self.best_nth_words} n={self.group_size} grade={self.grade:.2f}"

    def calculate_grade(self) -> float:
        return 0 + 0.5 * self.group_size + 2.0 * self.olympic_ratio

    @property
    def group_size(self) -> int:
        return len(self.best_nth_words)

    @property
    def detailed_string(self) -> str:
        return str(self.__dict__)


def normalize_vectors(u: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(u, axis=1)
    normalized = np.divide(u.T, norms).T
    return normalized


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between two word matrices.
    Each row in u and v is a word vector (the first dimension).
    :return: A cosine_similarities matrix, where cosine_similarities[i, j] = cosine_similarity(u[i], v[j])
    """
    u_normalized = normalize_vectors(u)
    v_normalized = normalize_vectors(v)
    return (u_normalized @ v_normalized.T).T


def get_card_color_index(card_color: CardColor) -> int:
    return {
        CardColor.BLUE: 0,
        CardColor.RED: 1,
        CardColor.GRAY: 2,
        CardColor.BLACK: 3,
    }[card_color]


SimilaritiesMatrix = np.ndarray
HeuristicsTensor = np.ndarray


class HeuristicsResult(NamedTuple):
    similarities: SimilaritiesMatrix
    heuristics: HeuristicsTensor


class HeuristicsCalculator:
    def __init__(
        self,
        board_words: List[str],
        model: KeyedVectors,
        current_heuristic: np.ndarray,
        team_card_color: CardColor,
        alpha: float,
        delta: float,
    ):
        self.board_words = board_words
        self.model = model
        self.current_heuristic = current_heuristic
        self.team_card_color = team_card_color
        self.alpha = alpha
        self.delta = delta

    def calculate_similarities_to_board(self, vocabulary: List[str]) -> np.ndarray:
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
        heuristics[i, j, k] = P(card[j].color = colors[k] | hint = vocabulary[i])
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


class ComplexProposalsGenerator:
    def __init__(
        self,
        model: KeyedVectors,
        model_adapter: ModelFormatAdapter,
        game_state: HinterGameState,
        team_card_color: CardColor,
        thresholds_filter_active: bool,
        current_heuristic: np.ndarray,
        alpha: float,
        delta: float,
        similarities: np.ndarray = None,
        vocabulary: Optional[List[str]] = None,
        most_common_ratio: float = 0.15,
        min_hint_frequency: float = 0.8,
        ratio_epsilon: float = 0.3,
    ):
        self.model = model
        self.model_adapter = model_adapter
        self.game_state = game_state
        self.team_card_color = team_card_color
        self.thresholds_filter_active = thresholds_filter_active
        self.current_heuristic = current_heuristic
        self.alpha = alpha
        self.delta = delta
        self.similarities = similarities
        self.vocabulary = vocabulary
        self.most_common_ratio = most_common_ratio
        self.min_hint_frequency = min_hint_frequency
        self.ratio_epsilon = ratio_epsilon

    def get_word_frequency(self, word: str) -> float:
        # TODO: len(self.model.key_to_index) is linear, this should not be linear grading.
        return 1 - self.model.key_to_index[word] / len(self.model.key_to_index)

    def model_format(self, word: str) -> str:
        return self.model_adapter.to_model_format(word)

    def board_format(self, word: str) -> str:
        return self.model_adapter.to_board_format(word)

    def should_filter_hint(self, hint: str, filter_expressions: Iterable[str]) -> bool:
        hint = self.board_format(hint)
        for filter_expression in filter_expressions:
            if hint in filter_expression or filter_expression in hint:
                return True
        if self.get_word_frequency(self.model_format(hint)) < self.min_hint_frequency:
            return True
        return False

    def filtered_vocabulary(self, vocabulary: List[str], filter_expressions: Iterable[str]) -> List[str]:
        return [word for word in vocabulary if not self.should_filter_hint(word, filter_expressions)]

    def generate_vocabulary(self) -> List[str]:
        amount_of_words = len(self.model.index_to_key) * self.most_common_ratio
        vocabulary = self.model.index_to_key[: int(amount_of_words)]
        filtered_vocabulary = self.filtered_vocabulary(
            vocabulary=vocabulary, filter_expressions=self.game_state.illegal_words
        )
        return list(filtered_vocabulary)

    @cached_property
    def board_words(self) -> List[str]:
        return [card.word for card in self.game_state.board]

    @cached_property
    def model_words(self) -> List[str]:
        return [self.model_format(word) for word in self.board_words]

    def generate_proposals(self) -> List[OlympicProposal]:
        vocabulary = self.vocabulary or self.generate_vocabulary()
        heuristics_calculator = HeuristicsCalculator(
            board_words=self.model_words,
            model=self.model,
            current_heuristic=self.current_heuristic,
            team_card_color=self.team_card_color,
            alpha=self.alpha,
            delta=self.delta,
        )
        similarities, heuristics = heuristics_calculator.calculate_heuristics_for_vocabulary(vocabulary)
        similarities_relu: SimilaritiesMatrix = np.maximum(similarities, 0)
        # heuristics.shape = (vocabulary_size, board_size, colors)
        # heuristics[i, j, k]= P(card[j].color = colors[k] | hint = vocabulary[i])
        unrevealed_mask = unrevealed_cards_mask(self.game_state.board)
        my_color_mask = card_color_mask(self.game_state.board, self.team_card_color)
        my_unrevealed_cards_mask = my_color_mask & unrevealed_mask
        other_unrevealed_cards_mask = ~my_color_mask & unrevealed_mask
        amount_of_my_cards = np.count_nonzero(my_unrevealed_cards_mask)
        vocab_indices = np.repeat(np.arange(len(vocabulary))[np.newaxis], repeats=amount_of_my_cards, axis=0).T
        my_color_index = get_card_color_index(self.team_card_color)
        my_cards_scores = heuristics.copy()[:, :, my_color_index]
        my_cards_scores[:, ~my_unrevealed_cards_mask] = -np.inf
        # Preform arg sort on scores, slice out -inf scores, reverse (high scores = low index)
        my_cards_idx_sorted = np.argsort(my_cards_scores)[:, : -amount_of_my_cards - 1 : -1]
        # my_cards_idx_sorted[i, j] = My j's top card similarity given hint word i
        my_nth_card_idx = my_cards_idx_sorted[:, :amount_of_my_cards]
        my_top_n_similarities = similarities_relu[vocab_indices, my_nth_card_idx]
        other_cards_similarities = similarities_relu[:, other_unrevealed_cards_mask]
        others_top_similarities_indices = np.argmax(other_cards_similarities, axis=1)
        others_top_similarities = other_cards_similarities[vocab_indices[:, 0], others_top_similarities_indices]
        olympia_ratio = np.divide(my_top_n_similarities.T, others_top_similarities + self.ratio_epsilon).T
        # olympia_ratio[i, j] = <vocab[i], best_cards[j]> / (<vocab[i], worst_card> + Ɛ)
        best_hints_indices = np.argmax(olympia_ratio, axis=0)
        proposals = []
        for size_index, hint_index in enumerate(best_hints_indices):
            group_size = size_index + 1
            olympic_ratio = olympia_ratio[hint_index, size_index]
            hint_word = vocabulary[hint_index]
            best_nth_indices = my_cards_idx_sorted[hint_index, :group_size]
            best_nth = tuple(self.board_words[i] for i in best_nth_indices)
            best_nth_similarities = my_top_n_similarities[hint_index, :group_size]
            worst_idx = others_top_similarities_indices[hint_index]
            worst = self.board_words[worst_idx]
            worst_similarity = other_cards_similarities[hint_index, worst_idx]
            proposal = OlympicProposal(
                hint_word=self.board_format(hint_word),
                best_nth_words=best_nth,
                best_nth_similarities=best_nth_similarities,
                worst=worst,
                worst_similarity=worst_similarity,
                olympic_ratio=olympic_ratio,
            )
            proposals.append(proposal)

        return proposals


class OlympicHinter(Hinter):
    def __init__(
        self,
        name: str,
        model: KeyedVectors = None,
        model_adapter: ModelFormatAdapter = None,
        gradual_distances_filter_active: bool = True,
        alpha: float = 4,
        delta: float = 0.1,
    ):
        super().__init__(name=name)
        self.model: KeyedVectors = model
        self.opponent_card_color = None
        self.model_adapter = model_adapter or DEFAULT_MODEL_ADAPTER
        self.gradual_distances_filter_active = gradual_distances_filter_active
        self.alpha = alpha
        self.delta = delta
        self.board_words: List[str] = []
        self.board_heuristic = np.array([])

    def on_game_start(self, language: str, board: Board):
        self.model = load_language(language=language)  # type: ignore
        self.opponent_card_color = self.team_color.opponent.as_card_color  # type: ignore
        self.board_words = [self.model_adapter.to_model_format(card.word) for card in board]
        self.board_heuristic = np.array([[0.25] * 4] * board.size)

    def model_format(self, word: str) -> str:
        return self.model_adapter.to_model_format(word)

    def on_hint_given(self, given_hint: GivenHint):
        try:
            heuristic_calculator = HeuristicsCalculator(
                board_words=self.board_words,
                model=self.model,
                current_heuristic=self.board_heuristic,
                team_card_color=given_hint.team_color.as_card_color,
                alpha=self.alpha,
                delta=self.delta,
            )
            _, updated_heuristics = heuristic_calculator.calculate_heuristics_for_vocabulary(
                vocabulary=[self.model_format(given_hint.word)]
            )
            self.board_heuristic = updated_heuristics[0]

        except Exception as e:
            log.warning(f"Hint {given_hint.word} not found in model, error: {e}")

    @classmethod
    def pick_best_proposal(cls, proposals: List[OlympicProposal]) -> OlympicProposal:
        log.debug(f"Got {len(proposals)} proposals.")
        if len(proposals) == 0:
            raise NoProposalsFound()
        for proposal in proposals:
            if proposal.group_size == 2:
                return proposal
        print_top_n = 5
        proposals.sort(key=lambda proposal: -proposal.grade)
        best_ton_n_repr = "\n".join(str(p) for p in proposals[:print_top_n])
        log.info(f"Best {print_top_n} proposals: \n{best_ton_n_repr}")
        best_proposal = proposals[0]
        log.debug(f"Picked proposal: {best_proposal.detailed_string}")
        return best_proposal

    def pick_hint(
        self,
        game_state: HinterGameState,
        thresholds_filter_active: bool = True,
    ) -> Hint:
        proposal_generator = ComplexProposalsGenerator(
            model=self.model,
            model_adapter=self.model_adapter,
            game_state=game_state,
            team_card_color=self.team_card_color,
            thresholds_filter_active=thresholds_filter_active,
            current_heuristic=self.board_heuristic,
            alpha=self.alpha,
            delta=self.delta,
            # vocabulary=self.model.index_to_key[:15000],
            # vocabulary=["אווירי"],
        )
        proposals = proposal_generator.generate_proposals()
        try:
            proposal = self.pick_best_proposal(proposals=proposals)
            word_group_board_format = tuple(
                self.model_adapter.to_board_format(word) for word in proposal.best_nth_words  # type: ignore
            )
            return Hint(proposal.hint_word, proposal.group_size, for_words=word_group_board_format)
        except NoProposalsFound:
            log.debug("No legal proposals found.")
            if not thresholds_filter_active:
                random_word = uuid4().hex[:4]
                return Hint(random_word, 1)
            log.info("Trying without thresholds filtering.")
            return self.pick_hint(game_state=game_state, thresholds_filter_active=False)

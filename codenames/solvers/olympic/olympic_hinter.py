import logging
from typing import Iterable, List, Optional, Tuple, no_type_check
from uuid import uuid4

import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm

from codenames.game import DEFAULT_MODEL_ADAPTER, Hinter, ModelFormatAdapter
from codenames.game.base import (
    Board,
    CardColor,
    GivenHint,
    Hint,
    HinterGameState,
    WordGroup,
)
from codenames.utils.async_task_manager import AsyncTaskManager
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


# min_frequency:         high = more results
# max_distance_group:     high = more results
# min_distance_gray:      high = fewer results
# min_distance_opponent:  high = fewer results
# min_distance_black:     high = fewer results
# @dataclass
# class ProposalThresholds:
#     min_frequency: float = 0  # Can't be less common than X.
#     max_distance_group: float = 1  # Can't be far from the group more than X.
#     min_distance_gray: float = 0  # Can't be closer to gray then X.
#     min_distance_opponent: float = 0  # Can't be closer to opponent then X.
#     min_distance_black: float = 0  # Can't be closer to black then X.
#
#     @staticmethod
#     def from_max_distance_group(max_distance_group: float) -> "ProposalThresholds":
#         return ProposalThresholds(
#             max_distance_group=max_distance_group,
#             min_distance_gray=max_distance_group * 1.04,
#             min_distance_opponent=max_distance_group * 1.14,
#             min_distance_black=max_distance_group * 1.18,
#         )


# class ThresholdDistances(dict):
#     @property
#     def min(self) -> float:
#         return min(self.values())
#
#     @property
#     def total(self) -> float:
#         return sum(self.values())
#
#
# DEFAULT_THRESHOLDS = ProposalThresholds(min_frequency=0.85, max_distance_group=0.30)


class OlympicProposal:
    def __init__(
        self,
        hint_word: str,
        best_nth: WordGroup = None,
        worst: str = None,
        grade: float = 0,
        **kwargs,
    ):
        self.hint_word = hint_word
        self.best_nth = best_nth
        self.worst = worst
        self.grade = grade
        self.extra = kwargs

    def __str__(self) -> str:
        return f"hint={self.hint_word} for={self.best_nth} n={self.group_size} grade={self.grade:.2f}"

    @property
    def group_size(self) -> int:
        return len(self.best_nth) if self.best_nth else -1

    @property
    def detailed_string(self) -> str:
        return str(self.__dict__)


# def group_is_closest(proposal: OlympicProposal) -> bool:
#     return all(
#         proposal.distance_group < other_distance
#         for other_distance in {proposal.distance_gray, proposal.distance_opponent, proposal.distance_black}
#     )


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

    def calculate_heuristics_for_vocabulary(self, vocabulary: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        similarities = self.calculate_similarities_to_board(vocabulary=vocabulary)
        return self.calculate_heuristics_for_similarities(similarities=similarities), similarities

    def calculate_heuristics_for_similarities(self, similarities: np.ndarray) -> np.ndarray:
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
        top_n: int = 5,
        most_common_ratio: float = 0.15,
        min_hint_frequency: float = 0.8,
        max_group_size: int = 4,
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
        self.top_n = top_n
        self.most_common_ratio = most_common_ratio
        self.min_hint_frequency = min_hint_frequency
        self.max_group_size = max_group_size
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
        return list(vocabulary)

    @property
    def unrevealed_words(self) -> List[str]:
        return [self.model_format(card.word) for card in self.game_state.board.unrevealed_cards]

    def generate_proposals(self) -> List[OlympicProposal]:
        vocabulary = self.vocabulary or self.generate_vocabulary()
        heuristics_calculator = HeuristicsCalculator(
            board_words=self.unrevealed_words,
            model=self.model,
            current_heuristic=self.current_heuristic,
            team_card_color=self.team_card_color,
            alpha=self.alpha,
            delta=self.delta,
        )
        heuristics, similarities = heuristics_calculator.calculate_heuristics_for_vocabulary(vocabulary)
        similarities_relu: np.ndarray = np.maximum(similarities, 0)
        # heuristics.shape = (vocabulary_size, board_size, colors)
        # heuristics[i, j, k] = P(card[j].color = colors[k] | hint = vocabulary[i])

        # Create card color and revealed masks
        my_color_index = get_card_color_index(self.team_card_color)
        other_colors_mask = np.ones(shape=(4,), dtype=bool)
        other_colors_mask[my_color_index] = False
        unrevealed_mask = unrevealed_cards_mask(self.game_state.board)
        my_color_mask = card_color_mask(self.game_state.board, self.team_card_color)
        my_unrevealed_cards_mask = my_color_mask & unrevealed_mask
        other_unrevealed_cards_mask = ~my_color_mask & unrevealed_mask

        # Create scores
        my_cards_scores = heuristics.copy()[:, :, my_color_index]
        # other_cards_scores = heuristics[:, :, other_colors_mask]
        # other_cards_scores = np.max(other_cards_scores, axis=2)
        my_cards_scores[:, ~my_unrevealed_cards_mask] = -np.inf
        # other_cards_scores[:, ~other_unrevealed_cards_mask] = -np.inf

        amount_of_my_cards = np.count_nonzero(my_unrevealed_cards_mask)
        # amount_of_other_cards = np.count_nonzero(other_unrevealed_cards_mask)

        # Preform arg sort on scores, slice out -inf scores, reverse (high scores = low index)
        my_cards_idx_sorted = np.argsort(my_cards_scores)[:, : -amount_of_my_cards - 1 : -1]
        # other_cards_idx_sorted = np.argsort(other_cards_scores)[:, : -amount_of_other_cards - 1: -1]
        # my_cards_idx_sorted[i, j] = My j's best card index given hint word i
        # other_cards_idx_sorted[i, j, k] = Other j's worst card index given hint word i for color k
        other_cards_similarities = similarities_relu[:, other_unrevealed_cards_mask]

        vocab_indices = np.repeat(np.arange(len(vocabulary))[np.newaxis], repeats=self.top_n, axis=0).T
        my_nth_card_idx = my_cards_idx_sorted[:, : self.top_n]
        my_top_n_similarities = similarities_relu[vocab_indices, my_nth_card_idx]

        others_top_similarities_indices = np.argmax(other_cards_similarities, axis=1)
        others_top_similarities = other_cards_similarities[vocab_indices[:, 0], others_top_similarities_indices]
        olympia_ratio = np.divide(my_top_n_similarities.T, others_top_similarities + self.ratio_epsilon).T
        best_hints_indices = np.argmax(olympia_ratio, axis=0)
        proposals = []
        for group_size, hint_index in enumerate(best_hints_indices):
            hint_word = vocabulary[hint_index]
            best_nth_indices = my_cards_idx_sorted[hint_index, : group_size + 1]
            best_nth = tuple(self.unrevealed_words[i] for i in best_nth_indices)
            worst_idx = others_top_similarities_indices[hint_index]
            worst = self.unrevealed_words[worst_idx]
            proposal = OlympicProposal(
                hint_word=self.board_format(hint_word),
                best_nth=best_nth,
                worst=worst,
            )
            proposals.append(proposal)

        return proposals

    @no_type_check
    def generate_proposals_2(self) -> List[OlympicProposal]:  # noqa
        vocabulary = self.vocabulary or self.generate_vocabulary()
        filtered_vocabulary = self.filtered_vocabulary(
            vocabulary=vocabulary, filter_expressions=self.game_state.illegal_words
        )

        my_color_scores = self.generate_possible_scores(filtered_vocabulary=filtered_vocabulary)

        revealed_mask = ~unrevealed_cards_mask(self.game_state.board)
        my_color_mask = card_color_mask(self.game_state.board, self.team_card_color)

        my_unrevealed_cards_mask = my_color_mask & ~revealed_mask
        other_unrevealed_cards_mask = ~my_color_mask & ~revealed_mask

        amount_of_my_cards = np.count_nonzero(my_unrevealed_cards_mask)
        amount_of_other_cards = np.count_nonzero(other_unrevealed_cards_mask)

        # Fill -inf where card color doesn't match or card is revealed
        my_cards_scores, other_cards_scores = my_color_scores.copy(), my_color_scores.copy()
        my_cards_scores[:, ~my_unrevealed_cards_mask] = -np.inf
        other_cards_scores[:, ~other_unrevealed_cards_mask] = -np.inf

        # Preform arg sort on scores, slice out -inf scores, reverse (high scores = low index)
        my_cards_idx_sorted = np.argsort(my_cards_scores)[:, : -amount_of_my_cards - 1 : -1]
        other_cards_idx_sorted = np.argsort(other_cards_scores)[:, : -amount_of_other_cards - 1 : -1]

        worst_card_idxs = [int(i) for i in other_cards_idx_sorted[:, 0]]
        worst_cards = [self.game_state.board[i] for i in worst_card_idxs]
        proposals = []
        for n in range(self.max_group_size):
            group_size = n + 1
            log.debug(f"Generating {group_size}-word group proposals...")
            best_nth_card_idxs = [int(i) for i in my_cards_idx_sorted[:, n]]
            best_nth_cards = [self.game_state.board[i] for i in best_nth_card_idxs]
            for i, (hint, best_card, worst_card) in enumerate(zip(filtered_vocabulary, best_nth_cards, worst_cards)):
                best_card_word, worst_card_word = self.model_format(best_card.word), self.model_format(worst_card.word)
                best_index, worst_index, hint_index = (
                    self.model.key_to_index[best_card_word],
                    self.model.key_to_index[worst_card_word],
                    self.model.key_to_index[hint],
                )
                good_similarity, bad_similarity = (
                    self.similarities[hint_index, best_index],
                    self.similarities[hint_index, worst_index],
                )
                if good_similarity < 0.1:
                    continue
                ratio = abs(good_similarity / (bad_similarity + self.ratio_epsilon))
                grade = ratio * group_size
                proposal = OlympicProposal(
                    hint_word=hint,
                    group_size=group_size,
                    best_nth=best_card.word,
                    worst=worst_card.word,
                    grade=grade,
                    my_scores=my_cards_scores[i],
                )
                proposals.append(proposal)
        return proposals

    def generate_possible_scores(self, filtered_vocabulary: List[str]) -> np.ndarray:
        log.debug("Generating possible scores...")
        task_manager = AsyncTaskManager()
        for hint in filtered_vocabulary:
            task_manager.add_task(
                self.current_heuristic.get_updated_board_heuristic,  # type: ignore
                args=(hint, self.team_card_color),
            )
        memories = list(tqdm(task_manager, total=len(filtered_vocabulary)))
        my_color_scores = np.array(
            [board_heuristic.get_scores_for_color(self.team_card_color) for board_heuristic in memories]
        )
        log.debug("Generated possible scores done.")
        return my_color_scores


class OlympicHinter(Hinter):
    def __init__(
        self,
        name: str,
        model: KeyedVectors = None,
        max_group_size: int = 4,
        model_adapter: ModelFormatAdapter = None,
        gradual_distances_filter_active: bool = True,
        alpha: float = 4,
        delta: float = 0.1,
    ):
        super().__init__(name=name)
        self.model: KeyedVectors = model
        self.max_group_size = max_group_size
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
                team_card_color=self.team_card_color,
                alpha=self.alpha,
                delta=self.delta,
            )
            updated_heuristic = heuristic_calculator.calculate_heuristics_for_vocabulary(
                vocabulary=[self.model_format(given_hint.word)]
            )
            self.board_heuristic = updated_heuristic[0]

        except Exception as e:
            log.warning(f"Hint {given_hint.word} not found in model, error: {e}")

    @classmethod
    def pick_best_proposal(cls, proposals: List[OlympicProposal]) -> OlympicProposal:
        log.debug(f"Got {len(proposals)} proposals.")
        if len(proposals) == 0:
            raise NoProposalsFound()
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
            max_group_size=self.max_group_size,
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
                self.model_adapter.to_board_format(word) for word in proposal.best_nth  # type: ignore
            )
            return Hint(proposal.hint_word, proposal.group_size, for_words=word_group_board_format)
        except NoProposalsFound:
            log.debug("No legal proposals found.")
            if not thresholds_filter_active:
                random_word = uuid4().hex[:4]
                return Hint(random_word, 1)
            log.info("Trying without thresholds filtering.")
            return self.pick_hint(game_state=game_state, thresholds_filter_active=False)

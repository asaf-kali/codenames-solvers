import logging
from typing import Iterable, List, Optional, Tuple
from uuid import uuid4

import editdistance
import numpy as np
from codenames.generic.board import Board, WordGroup
from codenames.generic.card import CardColor
from codenames.generic.move import Clue, GivenClue
from codenames.generic.player import Spymaster
from codenames.generic.state import SpymasterState
from gensim.models import KeyedVectors
from typing_extensions import TypeAlias

from codenames_solvers.models import (
    DEFAULT_MODEL_ADAPTER,
    ModelFormatAdapter,
    load_language,
)
from codenames_solvers.olympic.board_heuristics import (
    HeuristicsCalculator,
    HeuristicsTensor,
    SimilaritiesMatrix,
    get_card_color_index,
)

log = logging.getLogger(__name__)
Mask: TypeAlias = np.ndarray


class NoProposalsFound(Exception):
    pass


def unrevealed_cards_mask(board: Board) -> Mask:
    return np.array([not card.revealed for card in board])


def card_color_mask(board: Board, card_color: CardColor) -> Mask:
    idxs = [i for i, card in enumerate(board) if card.color == card_color]
    mask = np.zeros(board.size, dtype=bool)
    mask[idxs] = True
    return mask


class OlympicProposal:
    def __init__(
        self,
        clue_word: str,
        best_nth_words: WordGroup,
        best_nth_similarities: Tuple[float, ...],
        worst: str,
        worst_similarity: float,
        olympic_ratio: float,
        **kwargs,
    ):
        self.clue_word = clue_word
        self.best_nth_words = best_nth_words
        self.best_nth_similarities = best_nth_similarities
        self.worst = worst
        self.worst_similarity = worst_similarity
        self.olympic_ratio = olympic_ratio
        self.extra = kwargs
        self.grade = self.calculate_grade()

    def __str__(self) -> str:
        return f"clue={self.clue_word} for={self.best_nth_words} n={self.group_size} grade={self.grade:.2f}"

    def calculate_grade(self) -> float:
        optimal_size = 2.6
        size_distance = optimal_size - abs(optimal_size - self.group_size)
        grade = 1.1 * size_distance + 1.0 * self.olympic_ratio
        return grade

    @property
    def group_size(self) -> int:
        return len(self.best_nth_words)

    @property
    def detailed_string(self) -> str:
        return str(self.__dict__)


class ComplexProposalsGenerator:
    def __init__(
        self,
        model: KeyedVectors,
        model_adapter: ModelFormatAdapter,
        game_state: SpymasterState,
        team_card_color: CardColor,
        thresholds_filter_active: bool,
        current_heuristic: HeuristicsTensor,
        alpha: float,
        delta: float,
        vocabulary: List[str],
        similarities: Optional[SimilaritiesMatrix] = None,
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
        self.base_vocabulary = vocabulary
        self.ratio_epsilon = ratio_epsilon

    def model_format(self, word: str) -> str:
        return self.model_adapter.to_model_format(word)

    def board_format(self, word: str) -> str:
        return self.model_adapter.to_board_format(word)

    def get_updated_vocabulary(self) -> List[str]:
        filter_expressions = [self.model_format(word) for word in self.game_state.given_clue_words]
        return [word for word in self.base_vocabulary if word not in filter_expressions]

    @property
    def board_words(self) -> List[str]:
        return [card.word for card in self.game_state.board]

    @property
    def board_words_model_format(self) -> List[str]:
        return [self.model_format(word) for word in self.board_words]

    def generate_proposals(self) -> List[OlympicProposal]:
        vocabulary = self.get_updated_vocabulary()
        heuristics_calculator = HeuristicsCalculator(
            model=self.model,
            board_words=self.board_words_model_format,
            current_heuristic=self.current_heuristic,
            team_card_color=self.team_card_color,
            alpha=self.alpha,
            delta=self.delta,
        )
        similarities, heuristics = heuristics_calculator.calculate_heuristics_for_vocabulary(vocabulary)
        similarities_relu: SimilaritiesMatrix = np.maximum(similarities, 0)
        # heuristics.shape = (vocabulary_size, board_size, colors)
        # heuristics[i, j, k]= P(card[j].color = colors[k] | given_clue = vocabulary[i])
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
        # my_cards_idx_sorted[i, j] = My j's top card similarity given given_clue word i
        my_nth_card_idx = my_cards_idx_sorted[:, :amount_of_my_cards]
        my_top_n_similarities = similarities_relu[vocab_indices, my_nth_card_idx]
        other_cards_similarities = similarities_relu[:, other_unrevealed_cards_mask]
        others_top_similarities_indices = np.argmax(other_cards_similarities, axis=1)
        others_top_similarities = other_cards_similarities[vocab_indices[:, 0], others_top_similarities_indices]
        olympia_ratio = np.divide(my_top_n_similarities.T, others_top_similarities + self.ratio_epsilon).T
        # olympia_ratio[i, j] = <vocab[i], best_cards[j]> / (<vocab[i], worst_card> + Ɛ)
        best_clues_indices = np.argmax(olympia_ratio, axis=0)
        proposals = []
        for size_index, clue_index in enumerate(best_clues_indices):
            group_size = size_index + 1
            olympic_ratio = olympia_ratio[clue_index, size_index]
            clue_word = vocabulary[clue_index]
            best_nth_indices = my_cards_idx_sorted[clue_index, :group_size]
            best_nth = tuple(self.board_words[i] for i in best_nth_indices)
            best_nth_similarities = my_top_n_similarities[clue_index, :group_size]
            worst_idx = others_top_similarities_indices[clue_index]
            worst = self.board_words[worst_idx]
            worst_similarity = other_cards_similarities[clue_index, worst_idx]
            proposal = OlympicProposal(
                clue_word=self.board_format(clue_word),
                best_nth_words=best_nth,
                best_nth_similarities=best_nth_similarities,
                worst=worst,
                worst_similarity=worst_similarity,
                olympic_ratio=olympic_ratio,
            )
            proposals.append(proposal)

        return proposals


class VocabularyBuilder:
    def __init__(self, model: KeyedVectors, most_common_ratio: float, filter_expressions: Iterable[str]):
        self.model = model
        self.most_common_ratio = most_common_ratio
        self.filter_expressions = filter_expressions

    def build_vocabulary(self) -> List[str]:
        amount_of_words = int(len(self.model.index_to_key) * self.most_common_ratio)
        vocabulary = self.model.index_to_key[:amount_of_words]
        filtered_vocabulary = [word for word in vocabulary if not self.should_filter_clue(word)]
        return filtered_vocabulary

    def should_filter_clue(self, clue: str) -> bool:
        # l1 = len(given_clue)
        for expression in self.filter_expressions:
            # removals = get_removals_count(given_clue, expression)
            # longest = max(l1, l2)
            # diff = abs(l1 - l2)
            # l2 = len(expression)
            edit_distance = editdistance.eval(clue, expression)
            if len(clue) <= 4 or len(clue) <= 4:
                if edit_distance <= 1:
                    return True
            elif edit_distance <= 2:
                return True
        return False


class OlympicSpymaster(Spymaster):
    def __init__(
        self,
        name: str,
        model: KeyedVectors = None,
        model_adapter: Optional[ModelFormatAdapter] = None,
        gradual_distances_filter_active: bool = True,
        alpha: float = 4,
        delta: float = 0.1,
        most_common_ratio: float = 0.15,
    ):
        super().__init__(name=name)
        self.model: KeyedVectors = model
        self.opponent_card_color = None
        self.model_adapter = model_adapter or DEFAULT_MODEL_ADAPTER
        self.gradual_distances_filter_active = gradual_distances_filter_active
        self.alpha = alpha
        self.delta = delta
        self.most_common_ratio = most_common_ratio
        self.board_words: List[str] = []
        self.board_heuristic: HeuristicsTensor = np.array([])
        self.vocabulary: List[str] = []

    def on_game_start(self, board: Board):
        self.model = load_language(language=board.language)  # type: ignore
        self.opponent_card_color = self.team.opponent.as_card_color  # type: ignore
        self.board_words = [self.model_format(card.word) for card in board]
        self.board_heuristic = np.array([[0.25] * 4] * board.size)
        self.vocabulary = self.build_vocabulary()

    def give_clue(self, game_state: SpymasterState, thresholds_filter_active: bool = True) -> Clue:
        proposal_generator = ComplexProposalsGenerator(
            model=self.model,
            model_adapter=self.model_adapter,
            game_state=game_state,
            team_card_color=self.team_card_color,
            thresholds_filter_active=thresholds_filter_active,
            current_heuristic=self.board_heuristic,
            alpha=self.alpha,
            delta=self.delta,
            vocabulary=self.vocabulary,
            # vocabulary=["אווירי"],
        )
        proposals = proposal_generator.generate_proposals()
        try:
            proposal = self.pick_best_proposal(proposals=proposals)
            word_group_board_format = tuple(
                self.model_adapter.to_board_format(word) for word in proposal.best_nth_words  # type: ignore
            )
            return Clue(word=proposal.clue_word, card_amount=proposal.group_size, for_words=word_group_board_format)
        except NoProposalsFound:
            log.warning("No legal proposals found.")
            if not thresholds_filter_active:
                random_word = uuid4().hex[:4]
                return Clue(word=random_word, card_amount=1)
            log.info("Trying without thresholds filtering.")
            return self.give_clue(game_state=game_state, thresholds_filter_active=False)

    def on_clue_given(self, given_clue: GivenClue):
        try:
            heuristic_calculator = HeuristicsCalculator(
                model=self.model,
                board_words=self.board_words,
                current_heuristic=self.board_heuristic,
                team_card_color=given_clue.team.as_card_color,
                alpha=self.alpha,
                delta=self.delta,
            )
            _, updated_heuristics = heuristic_calculator.calculate_heuristics_for_vocabulary(
                vocabulary=[self.model_format(given_clue.word)]
            )
            self.board_heuristic = updated_heuristics[0]

        except Exception as e:
            log.warning(f"Clue {given_clue.word} not found in model, error: {e}")

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

    def model_format(self, word: str) -> str:
        return self.model_adapter.to_model_format(word)

    def build_vocabulary(self) -> List[str]:
        vocabulary_builder = VocabularyBuilder(
            model=self.model, most_common_ratio=self.most_common_ratio, filter_expressions=self.board_words
        )
        return vocabulary_builder.build_vocabulary()

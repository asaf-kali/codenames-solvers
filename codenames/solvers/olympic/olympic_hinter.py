import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Iterable, List, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from tqdm import tqdm

from codenames.game import DEFAULT_MODEL_ADAPTER, Hinter, ModelFormatAdapter
from codenames.game.base import (
    Board,
    CardColor,
    CardSet,
    GivenHint,
    Hint,
    HinterGameState,
    WordGroup,
)
from codenames.solvers.olympic.memory import Memory
from codenames.solvers.utils.algebra import cosine_distance, cosine_similarity
from codenames.utils.async_task_manager import AsyncTaskManager
from language_data.model_loader import load_language

log = logging.getLogger(__name__)


class NoProposalsFound(Exception):
    pass


# min_frequency:         high = more results
# max_distance_group:     high = more results
# min_distance_gray:      high = fewer results
# min_distance_opponent:  high = fewer results
# min_distance_black:     high = fewer results
@dataclass
class ProposalThresholds:
    min_frequency: float = 0  # Can't be less common than X.
    max_distance_group: float = 1  # Can't be far from the group more than X.
    min_distance_gray: float = 0  # Can't be closer to gray then X.
    min_distance_opponent: float = 0  # Can't be closer to opponent then X.
    min_distance_black: float = 0  # Can't be closer to black then X.

    @staticmethod
    def from_max_distance_group(max_distance_group: float) -> "ProposalThresholds":
        return ProposalThresholds(
            max_distance_group=max_distance_group,
            min_distance_gray=max_distance_group * 1.04,
            min_distance_opponent=max_distance_group * 1.14,
            min_distance_black=max_distance_group * 1.18,
        )


class ThresholdDistances(dict):
    @property
    def min(self) -> float:
        return min(self.values())

    @property
    def total(self) -> float:
        return sum(self.values())


DEFAULT_THRESHOLDS = ProposalThresholds(min_frequency=0.85, max_distance_group=0.30)


class OlympicProposal:
    def __init__(
        self,
        hint_word: str,
        n: int = None,
        best_nth: str = None,
        worst: str = None,
        word_group: WordGroup = None,
        grade: float = 0,
        **kwargs,
    ):
        self.hint_word = hint_word
        self.n = n
        self.best_nth = best_nth
        self.worst = worst
        self.word_group = word_group or tuple()
        self.grade = grade
        self.extra = kwargs

    def __str__(self) -> str:
        return f"hint={self.hint_word} for={self.best_nth} n={self.n} grade={self.grade:.2f}"

    @property
    def card_count(self) -> int:
        return self.n or 0

    @property
    def detailed_string(self) -> str:
        return str(self.__dict__)


# def group_is_closest(proposal: OlympicProposal) -> bool:
#     return all(
#         proposal.distance_group < other_distance
#         for other_distance in {proposal.distance_gray, proposal.distance_opponent, proposal.distance_black}
#     )

SimilarityKey = Tuple[str, str]


class SimilarityCache:
    def __init__(self, model: KeyedVectors):
        self._cache: Dict[SimilarityKey, float] = {}
        self._model = model

    def __getitem__(self, key: SimilarityKey) -> float:
        return self.similarity(*key)

    def __setitem__(self, key: SimilarityKey, value: float):
        self._cache[(key[0], key[1])] = value
        self._cache[(key[1], key[0])] = value

    def similarity(self, word1: str, word2: str) -> float:
        try:
            return self._cache[(word1, word2)]
        except KeyError:
            similarity = self._model.similarity(word1, word2)
            self[(word1, word2)] = similarity
            return similarity


@dataclass
class ComplexProposalsGenerator:
    model: KeyedVectors
    model_adapter: ModelFormatAdapter
    game_state: HinterGameState
    team_card_color: CardColor
    proposals_thresholds: ProposalThresholds
    # proposal_grade_calculator: Callable[[OlympicProposal], float]
    thresholds_filter_active: bool
    gradual_distances_filter_active: bool
    similarities_top_n: int
    max_group_size: int
    memory: Memory
    vocabulary: List[str]

    def __post_init__(self):
        unrevealed_cards = self.game_state.board.unrevealed_cards
        words = tuple(self.model_format(card.word) for card in unrevealed_cards)
        colors = tuple(card.color for card in unrevealed_cards)
        # vectors_as_lists_list: List[List[float]] = self.model[words].tolist()  # type: ignore
        # vectors_list = [np.array(v) for v in vectors_as_lists_list]
        vectors_list = [v for v in self.model[words]]
        self.board_data = pd.DataFrame(
            data={
                "color": colors,
                "vectors": vectors_list,
            },
            index=words,
        )

    def get_vectors(self, index: np.ndarray) -> pd.Series:
        return self.board_data[index]["vectors"]

    def word_group_vectors(self, word_group: WordGroup) -> pd.Series:
        return self.get_vectors(self.board_data.index.isin(word_group))

    @cached_property
    def team_unrevealed_cards(self) -> CardSet:
        return self.game_state.board.unrevealed_cards_for_color(self.team_card_color)

    @cached_property
    def gray_vectors(self) -> pd.Series:
        return self.get_vectors(self.board_data.color == CardColor.GRAY)

    @cached_property
    def opponent_vectors(self) -> pd.Series:
        return self.get_vectors(self.board_data.color == self.team_card_color.opponent)

    @cached_property
    def black_vectors(self) -> pd.Series:
        return self.get_vectors(self.board_data.color == CardColor.BLACK)

    def get_word_frequency(self, word: str) -> float:
        # TODO: len(self.model.key_to_index) is linear, this should not be linear grading.
        return 1 - self.model.key_to_index[word] / len(self.model.key_to_index)

    def model_format(self, word: str) -> str:
        return self.model_adapter.to_model_format(word)

    def board_format(self, word: str) -> str:
        return self.model_adapter.to_board_format(word)

    # def proposal_satisfy_thresholds(self, proposal: OlympicProposal) -> bool:
    #     if not self.thresholds_filter_active:
    #         return True
    #     distances = ThresholdDistances(
    #         frequency=proposal.hint_word_frequency - self.proposals_thresholds.min_frequency,
    #         group=self.proposals_thresholds.max_distance_group - proposal.distance_group,
    #         gray=proposal.distance_gray - self.proposals_thresholds.min_distance_gray,
    #         opponent=proposal.distance_opponent - self.proposals_thresholds.min_distance_opponent,
    #         black=proposal.distance_black - self.proposals_thresholds.min_distance_black,
    #     )
    #     pass_thresholds = distances.min >= 0
    #     really_good = distances.min >= -0.05 and distances.total >= 0.45
    #     return pass_thresholds or really_good

    def should_filter_hint(self, hint: str, filter_expressions: Iterable[str]) -> bool:
        # if "_" in word:
        #     return True
        # if word in BANNED_WORDS:
        #     return True
        hint = self.board_format(hint)
        for filter_expression in filter_expressions:
            if hint in filter_expression or filter_expression in hint:
                return True
        # for word in word_group:
        #     edit_distance = editdistance.eval(hint, word)
        #     if len(word) <= 4 or len(hint) <= 4:
        #         if edit_distance <= 1:
        #             return True
        #     elif edit_distance <= 2:
        #         return True
        return False

    def filtered_vocabulary(self, filter_expressions: Iterable[str]) -> List[str]:
        return [word for word in self.vocabulary if not self.should_filter_hint(word, filter_expressions)]

    def init_similarity_cache(self, filtered_vocabulary: List[str]) -> SimilarityCache:
        log.debug(f"Initializing similarity cache with {len(filtered_vocabulary)} words...")
        board_words_model_format = [self.model_format(word) for word in self.game_state.board.all_words]
        board_vectors = np.array(list(self.model[word] for word in board_words_model_format))
        vocabulary_vectors = np.array(self.model[filtered_vocabulary])
        cosine_similarities: np.ndarray = cosine_similarity(board_vectors.T, vocabulary_vectors.T)  # type: ignore
        similarity_cache = SimilarityCache(self.model)
        for i, j in np.ndindex(cosine_similarities.shape):  # type: ignore
            word1, word2 = board_words_model_format[i], filtered_vocabulary[j]
            similarity_cache[word1, word2] = cosine_similarities[i, j]
        log.debug("Initialized similarity cache done.")
        return similarity_cache

    def generate_proposals(self) -> List[OlympicProposal]:
        filtered_vocabulary = self.filtered_vocabulary(filter_expressions=self.game_state.illegal_words)
        similarity_cache = self.init_similarity_cache(filtered_vocabulary)

        my_color_scores = self.generate_possible_scores(filtered_vocabulary)

        revealed_mask = self.game_state.board.revealed_cards_mask()
        my_color_mask = self.game_state.board.card_color_mask(self.team_card_color)

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
                ratio = similarity_cache[hint, best_card_word] / similarity_cache[hint, worst_card_word]
                proposal = OlympicProposal(
                    hint_word=hint,
                    n=group_size,
                    best_nth=best_card.word,
                    worst=worst_card.word,
                    grade=float(ratio) * group_size,
                    my_scores=my_cards_scores[i],
                )
                proposals.append(proposal)
        return proposals

    def generate_possible_scores(self, filtered_vocabulary: List[str]) -> np.ndarray:
        log.debug("Generating possible scores...")
        task_manager = AsyncTaskManager()
        for hint in filtered_vocabulary:
            task_manager.add_task(self.memory.get_updated_memory, args=(hint, self.team_card_color))
        memories = list(tqdm(task_manager, total=len(filtered_vocabulary)))
        my_color_scores = np.array([memory.get_scores_for_color(self.team_card_color) for memory in memories])
        log.debug("Generated possible scores done.")
        return my_color_scores

    # def local_maximum(self, centroid: np.ndarray, n: int):
    #     my_relevant_cards = self.game_state.board.unrevealed_cards_for_color(self.team_card_color)
    #     my_relevant_words = [self.model_format(card.word) for card in my_relevant_cards]
    #     my_relevant_vectors = self.model[my_relevant_words]
    #     # numerator = self.model.cosine_similarities(centroid, my_relevant_vectors)
    #     pass

    def is_cleared_for_proposal(self, group_vectors, centroid) -> bool:
        centroid_to_group = cosine_distance(centroid, group_vectors)
        max_centroid_to_group = np.max(centroid_to_group)

        wrong_cards = self.game_state.board.unrevealed_cards
        wrong_cards = wrong_cards.difference(self.game_state.board.unrevealed_cards_for_color(self.team_card_color))
        wrong_words = tuple(self.model_format(card.word) for card in wrong_cards)
        wrong_vectors = self.word_group_vectors(wrong_words)
        centroid_to_wrong_group = cosine_distance(centroid, wrong_vectors)
        min_centroid_to_group = np.min(centroid_to_wrong_group)

        return max_centroid_to_group < min_centroid_to_group


class OlympicHinter(Hinter):
    def __init__(
        self,
        name: str,
        model: KeyedVectors = None,
        proposals_thresholds: ProposalThresholds = None,
        max_group_size: int = 4,
        model_adapter: ModelFormatAdapter = None,
        gradual_distances_filter_active: bool = True,
        # proposal_grade_calculator: Callable[[OlympicProposal], float] = default_proposal_grade_calculator,
    ):
        super().__init__(name=name)
        self.model = model
        self.max_group_size = max_group_size
        self.opponent_card_color = None
        self.proposals_thresholds = proposals_thresholds or DEFAULT_THRESHOLDS
        self.model_adapter = model_adapter or DEFAULT_MODEL_ADAPTER
        self.gradual_distances_filter_active = gradual_distances_filter_active
        # self.proposal_grade_calculator = proposal_grade_calculator
        self.memory: Memory = None  # type: ignore

    def on_game_start(self, language: str, board: Board):
        self.model = load_language(language=language)  # type: ignore
        self.opponent_card_color = self.team_color.opponent.as_card_color  # type: ignore
        self.memory = Memory(alpha=4, delta=0.1, model=self.model, board=board)  # type: ignore

    def on_hint_given(self, given_hint: GivenHint):
        self.memory = self.memory.get_updated_memory(
            word=given_hint.word, team_color=given_hint.team_color.as_card_color
        )

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
        self, game_state: HinterGameState, thresholds_filter_active: bool = True, similarities_top_n: int = 10
    ) -> Hint:
        proposal_generator = ComplexProposalsGenerator(
            model=self.model,
            model_adapter=self.model_adapter,
            game_state=game_state,
            team_card_color=self.team_color.as_card_color,  # type: ignore
            proposals_thresholds=self.proposals_thresholds,
            # proposal_grade_calculator=self.proposal_grade_calculator,
            thresholds_filter_active=thresholds_filter_active,
            gradual_distances_filter_active=self.gradual_distances_filter_active,
            similarities_top_n=similarities_top_n,
            max_group_size=self.max_group_size,
            memory=self.memory,  # type: ignore
            vocabulary=self.model.index_to_key[:1000],  # type: ignore
        )
        proposals = proposal_generator.generate_proposals()
        try:
            proposal = self.pick_best_proposal(proposals=proposals)
            word_group_board_format = tuple(self.model_adapter.to_board_format(word) for word in proposal.word_group)
            return Hint(proposal.hint_word, proposal.card_count, for_words=word_group_board_format)
        except NoProposalsFound:
            log.debug("No legal proposals found.")
            if not thresholds_filter_active and similarities_top_n >= 20:
                random_word = uuid4().hex[:4]
                return Hint(random_word, 1)
            log.info("Trying without thresholds filtering.")
            return self.pick_hint(game_state=game_state, thresholds_filter_active=False, similarities_top_n=50)

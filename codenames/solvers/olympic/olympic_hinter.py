import itertools
import json
import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, List
from uuid import uuid4

import editdistance as editdistance
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from codenames.game import DEFAULT_MODEL_ADAPTER, Hinter, ModelFormatAdapter
from codenames.game.base import (
    Board,
    CardColor,
    CardSet,
    GivenHint,
    Hint,
    HinterGameState,
    Similarity,
    WordGroup,
)
from codenames.solvers.olympic.memory import Memory
from codenames.solvers.utils.algebra import cosine_distance
from codenames.utils import wrap
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


@dataclass
class OlympicProposal:
    word_group: WordGroup
    hint_word: str
    n: int
    r: str
    b_n: str
    grade: float = 0

    def __str__(self) -> str:
        return f"{self.word_group} = ('{self.hint_word}', {self.grade:.2f})"

    @property
    def card_count(self) -> int:
        return len(self.word_group)

    @property
    def detailed_string(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


# def group_is_closest(proposal: OlympicProposal) -> bool:
#     return all(
#         proposal.distance_group < other_distance
#         for other_distance in {proposal.distance_gray, proposal.distance_opponent, proposal.distance_black}
#     )


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

    def should_filter_hint(self, hint: str, word_group: WordGroup, filter_expressions: Iterable[str]) -> bool:
        # if "_" in word:
        #     return True
        # if word in BANNED_WORDS:
        #     return True
        hint = self.board_format(hint)
        for filter_expression in filter_expressions:
            if hint in filter_expression or filter_expression in hint:
                return True
        for word in word_group:
            edit_distance = editdistance.eval(hint, word)
            if len(word) <= 4 or len(hint) <= 4:
                if edit_distance <= 1:
                    return True
            elif edit_distance <= 2:
                return True
        return False

    def proposal_from_similarity(self, word_group: WordGroup, similarity: Similarity) -> List[OlympicProposal]:
        hint, similarity_score = similarity
        # word = format_word(word)
        if self.should_filter_hint(hint=hint, word_group=word_group, filter_expressions=self.game_state.illegal_words):
            return []
        # hint_vector = self.model[hint]
        if self.get_word_frequency(hint) < 0.85:
            return []
        memory = self.memory.get_updated_memory(hint, self.team_card_color)
        scores = memory.get_scores_for_color(self.team_card_color)
        unrevealed_card_mask = self.game_state.board.mask_of_unrevealed_cards_for_color(self.team_card_color)
        bad_cards_idxs_sorted = np.argsort(scores[~unrevealed_card_mask])
        r_i = int(bad_cards_idxs_sorted[0])
        r = self.game_state.board[r_i]
        my_cards_idxs_sorted = np.argsort(scores[unrevealed_card_mask])
        proposals = []
        for n in range(self.max_group_size):
            b_n_i = int(my_cards_idxs_sorted[n - 1])
            b_n = self.game_state.board[b_n_i]
            f_hint_n = self.model.similarity(hint, b_n.word) / self.model.similarity(hint, r.word)
            proposal = OlympicProposal(
                word_group=word_group,
                hint_word=hint,
                n=n,
                r=r.word,
                b_n=b_n.word,
                grade=float(f_hint_n),
            )
            proposals.append(proposal)
        # if self.gradual_distances_filter_active and not group_is_closest(proposal=proposal):
        #     return None
        # if not self.proposal_satisfy_thresholds(proposal=proposal):
        #     return None
        # proposal.grade = self.proposal_grade_calculator(proposal)
        return proposals

    def create_proposals_from_similarities(
        self, word_group: WordGroup, similarities: List[Similarity]
    ) -> List[OlympicProposal]:
        proposals = []
        for similarity in similarities:
            similarity_proposals = self.proposal_from_similarity(word_group=word_group, similarity=similarity)
            proposals.extend(similarity_proposals)
        return proposals

    def create_proposals_for_word_group(self, word_group: WordGroup) -> List[OlympicProposal]:
        # log.debug(f"Creating proposals for group: {word_group}.")
        vectors = self.model[word_group]  # type: ignore
        centroid = np.mean(vectors, axis=0)
        group_vectors = self.word_group_vectors(word_group)
        if self.is_cleared_for_proposal(group_vectors, centroid):
            similarities = self.model.most_similar(centroid, topn=self.similarities_top_n)  # type: ignore
            return self.create_proposals_from_similarities(word_group=word_group, similarities=similarities)
        return []

    def create_proposals_for_group_size(self, group_size: int) -> List[OlympicProposal]:
        log.debug(f"Creating proposals for group size {wrap(group_size)}...")
        proposals = []
        task_manager = AsyncTaskManager()
        for card_group in itertools.combinations(self.team_unrevealed_cards, group_size):
            word_group = tuple(self.model_format(card.word) for card in card_group)
            task_manager.add_task(self.create_proposals_for_word_group, args=(word_group,))
        log.debug("Waiting for task manager to finish...")
        for result in task_manager:
            proposals.extend(result)
        return proposals

    def generate_proposals(self, max_group_size: int):
        proposals = []
        for group_size in range(max_group_size, 0, -1):
            group_size_proposals = self.create_proposals_for_group_size(group_size=group_size)
            proposals.extend(group_size_proposals)
        return proposals

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
        max_group_size: int = 3,
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
        )
        proposals = proposal_generator.generate_proposals(self.max_group_size)
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

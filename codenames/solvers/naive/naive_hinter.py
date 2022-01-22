import itertools
import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from codenames.game import DEFAULT_MODEL_ADAPTER, Hinter, ModelFormatAdapter
from codenames.game.base import (
    Board,
    CardColor,
    CardSet,
    Hint,
    HinterGameState,
    Similarity,
    WordGroup,
)
from codenames.solvers.utils.algebra import cosine_distance
from codenames.utils import wrap
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
    min_frequency: float = 0.85  # Can't be less common than X.
    max_distance_group: float = 0.25  # Can't be far from the group more than X.
    min_distance_gray: float = 0.26  # Can't be closer to gray then X.
    min_distance_opponent: float = 0.28  # Can't be closer to opponent then X.
    min_distance_black: float = 0.30  # Can't be closer to black then X.


class ThresholdDistances(dict):
    @property
    def min(self) -> float:
        return min(self.values())

    @property
    def total(self) -> float:
        return sum(self.values())


DEFAULT_THRESHOLDS = ProposalThresholds()


@dataclass
class Proposal:
    word_group: WordGroup
    hint_word: str
    hint_word_frequency: float
    distance_group: float
    distance_gray: float
    distance_opponent: float
    distance_black: float
    grade: float = 0

    def __str__(self) -> str:
        return f"{self.word_group} = ('{self.hint_word}', {self.grade:.2f})"

    @property
    def card_count(self) -> int:
        return len(self.word_group)

    @property
    def detailed_string(self) -> str:
        return (
            f"Proposal(word_group={self.word_group}, "
            f"hint_word={self.hint_word}, "
            f"hint_word_frequency={self.hint_word_frequency:.3f}, "
            f"distance_group={self.distance_group:.3f}, "
            f"distance_gray={self.distance_gray:.3f}, "
            f"distance_opponent={self.distance_opponent:.3f}, "
            f"distance_black={self.distance_black:.3f}, "
            f"grade={self.grade:.3f})"
        )


def calculate_proposal_grade(proposal: Proposal) -> float:
    """
    High grade is good.
    """
    grade = (
        1.5 * len(proposal.word_group)
        - 2.5 * proposal.distance_group
        + 1.0 * proposal.distance_gray
        + 1.5 * proposal.distance_opponent
        + 2.0 * proposal.distance_black
    )
    return float(np.nan_to_num(grade, nan=-100))


@dataclass
class NaiveProposalsGenerator:
    model: KeyedVectors
    model_adapter: ModelFormatAdapter
    game_state: HinterGameState
    proposals_thresholds: ProposalThresholds
    team_card_color: CardColor
    thresholds_filter_active: bool

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

    def proposal_satisfy_thresholds(self, proposal: Proposal) -> bool:
        if not self.thresholds_filter_active:
            return True
        distances = ThresholdDistances(
            frequency=proposal.hint_word_frequency - self.proposals_thresholds.min_frequency,
            group=self.proposals_thresholds.max_distance_group - proposal.distance_group,
            gray=proposal.distance_gray - self.proposals_thresholds.min_distance_gray,
            opponent=proposal.distance_opponent - self.proposals_thresholds.min_distance_opponent,
            black=proposal.distance_black - self.proposals_thresholds.min_distance_black,
        )
        pass_thresholds = distances.min >= 0
        really_good = distances.min >= -0.01 and distances.total >= 0.45
        return pass_thresholds or really_good

    def should_filter_hint(self, word: str, filter_expressions: Iterable[str]) -> bool:
        # if "_" in word:
        #     return True
        # if word in BANNED_WORDS:
        #     return True
        word = self.board_format(word)
        for filter_expression in filter_expressions:
            if word in filter_expression or filter_expression in word:
                return True
        return False

    def proposal_from_similarity(self, word_group: WordGroup, similarity: Similarity) -> Optional[Proposal]:
        hint, similarity_score = similarity
        # word = format_word(word)
        if self.should_filter_hint(word=hint, filter_expressions=self.game_state.illegal_words):
            return None
        hint_vector = self.model[hint]
        hint_to_group = cosine_distance(hint_vector, self.word_group_vectors(word_group))
        hint_to_gray = cosine_distance(hint_vector, self.gray_vectors)
        hint_to_opponent = cosine_distance(hint_vector, self.opponent_vectors)
        hint_to_black = cosine_distance(hint_vector, self.black_vectors)
        proposal = Proposal(
            word_group=word_group,
            hint_word=self.board_format(hint),
            hint_word_frequency=self.get_word_frequency(hint),
            distance_group=np.max(hint_to_group),
            distance_gray=np.min(hint_to_gray),
            distance_opponent=np.min(hint_to_opponent),
            distance_black=np.min(hint_to_black),
        )
        if not self.proposal_satisfy_thresholds(proposal=proposal):
            return None
        proposal.grade = calculate_proposal_grade(proposal)
        return proposal

    def create_proposals_from_similarities(
        self, word_group: WordGroup, similarities: List[Similarity]
    ) -> List[Proposal]:
        proposals = []
        for similarity in similarities:
            proposal = self.proposal_from_similarity(word_group=word_group, similarity=similarity)
            if proposal is not None:
                proposals.append(proposal)
        return proposals

    def create_proposals_for_word_group(self, word_group: WordGroup) -> List[Proposal]:
        # log.debug(f"Creating proposals for group: {word_group}.")
        vectors = self.model[word_group]  # type: ignore
        centroid = np.mean(vectors, axis=0)
        group_vectors = self.word_group_vectors(word_group)
        centroid_to_group = cosine_distance(centroid, group_vectors)
        max_centroid_to_group = np.max(centroid_to_group)
        if self.thresholds_filter_active and max_centroid_to_group > self.proposals_thresholds.max_distance_group:
            return []
        # distances = cosine_distance(centroid, vectors)
        # group_similarity = np.mean(distances)
        similarities = self.model.most_similar(centroid)  # type: ignore
        return self.create_proposals_from_similarities(word_group=word_group, similarities=similarities)

    def create_proposals_for_group_size(self, group_size: int) -> List[Proposal]:
        log.info(f"Creating proposals for group size {wrap(group_size)}...")
        proposals = []
        for card_group in itertools.combinations(self.team_unrevealed_cards, group_size):
            word_group = tuple(self.model_format(card.word) for card in card_group)
            word_group_proposals = self.create_proposals_for_word_group(word_group=word_group)
            proposals.extend(word_group_proposals)
        return proposals

    def generate_proposals(self, max_group_size: int):
        proposals = []
        for group_size in range(max_group_size, 0, -1):
            group_size_proposals = self.create_proposals_for_group_size(group_size=group_size)
            proposals.extend(group_size_proposals)
        return proposals


class NaiveHinter(Hinter):
    def __init__(
        self,
        name: str,
        model: KeyedVectors = None,
        proposals_thresholds: ProposalThresholds = DEFAULT_THRESHOLDS,
        max_group_size: int = 3,
        model_adapter: ModelFormatAdapter = DEFAULT_MODEL_ADAPTER,
    ):
        super().__init__(name=name)
        self.model: KeyedVectors = model
        self.max_group_size = max_group_size
        self.opponent_card_color = None
        self.proposals_thresholds = proposals_thresholds
        self.model_adapter = model_adapter

    def on_game_start(self, language: str, board: Board):
        self.model = load_language(language=language)  # type: ignore
        self.opponent_card_color = self.team_color.opponent.as_card_color  # type: ignore

    @classmethod
    def pick_best_proposal(cls, proposals: List[Proposal]) -> Proposal:
        log.debug(f"Got {len(proposals)} proposals.")
        if len(proposals) == 0:
            raise NoProposalsFound()
        proposals.sort(key=lambda proposal: -proposal.grade)
        best_5_repr = "\n".join(str(p) for p in proposals[:5])
        log.info(f"Best 5 proposals: \n{best_5_repr}")
        best_proposal = proposals[0]
        log.info(f"Picked proposal: {best_proposal.detailed_string}")
        return best_proposal

    def pick_hint(self, game_state: HinterGameState, thresholds_filter_active: bool = True) -> Hint:
        proposal_generator = NaiveProposalsGenerator(
            model=self.model,
            model_adapter=self.model_adapter,
            game_state=game_state,
            proposals_thresholds=self.proposals_thresholds,
            team_card_color=self.team_color.as_card_color,  # type: ignore
            thresholds_filter_active=thresholds_filter_active,
        )
        proposals = proposal_generator.generate_proposals(self.max_group_size)
        try:
            proposal = self.pick_best_proposal(proposals=proposals)
            return Hint(proposal.hint_word, proposal.card_count, for_words=proposal.word_group)
        except NoProposalsFound:
            log.warning("No legal proposals found.")
            if not thresholds_filter_active:
                random_word = uuid4().hex[:4]
                return Hint(random_word, 2)
            log.info("Trying without thresholds filtering.")
            return self.pick_hint(game_state=game_state, thresholds_filter_active=False)

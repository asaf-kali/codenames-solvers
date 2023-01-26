import itertools
import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import editdistance as editdistance
import numpy as np
import pandas as pd
from codenames.game import (
    Board,
    CardColor,
    Cards,
    Hint,
    Hinter,
    HinterGameState,
    WordGroup,
)
from gensim.models import KeyedVectors
from pydantic import BaseModel
from the_spymaster_util.async_task_manager import AsyncTaskManager
from the_spymaster_util.logger import wrap

from solvers.models import (
    DEFAULT_MODEL_ADAPTER,
    ModelFormatAdapter,
    ModelIdentifier,
    load_language,
    load_model,
)
from solvers.utils.algebra import cosine_distance

log = logging.getLogger(__name__)

Similarity = Tuple[str, float]


class NoProposalsFound(Exception):
    pass


# min_frequency:         high = more results
# max_distance_group:     high = more results
# min_distance_gray:      high = fewer results
# min_distance_opponent:  high = fewer results
# min_distance_black:     high = fewer results
class ProposalThresholds(BaseModel):
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


DEFAULT_THRESHOLDS = ProposalThresholds(min_frequency=0.82, max_distance_group=0.27, min_distance_black=0.43)


class Proposal(BaseModel):
    word_group: WordGroup
    hint_word: str
    hint_word_frequency: float
    distance_group: float
    distance_gray: float
    distance_opponent: float
    distance_black: float
    grade: float = 0
    board_distances: Dict[str, float] = {}

    def __str__(self) -> str:
        return f"{self.word_group} = ('{self.hint_word}', {self.grade:.2f})"

    @property
    def card_count(self) -> int:
        return len(self.word_group)

    def dict(self, *args, **kwargs):
        result = super().dict(*args, **kwargs)
        _format_dict_floats(result)
        return result


def default_proposal_grade_calculator(proposal: Proposal) -> float:
    """
    High grade is good.
    """
    grade = (
        1.6 * len(proposal.word_group)
        + 1.8 * proposal.hint_word_frequency
        - 3.5 * proposal.distance_group  # High group distance is bad.
        + 1.0 * proposal.distance_gray
        + 2.0 * proposal.distance_opponent
        + 3.0 * proposal.distance_black
    )
    return float(np.nan_to_num(grade, nan=-100))


def group_is_closest(proposal: Proposal) -> bool:
    return all(
        proposal.distance_group < other_distance
        for other_distance in {proposal.distance_gray, proposal.distance_opponent, proposal.distance_black}
    )


@dataclass
class NaiveProposalsGenerator:
    model: KeyedVectors
    model_adapter: ModelFormatAdapter
    game_state: HinterGameState
    team_card_color: CardColor
    proposals_thresholds: ProposalThresholds
    proposal_grade_calculator: Callable[[Proposal], float]
    thresholds_filter_active: bool
    gradual_distances_filter_active: bool
    similarities_top_n: int

    def __post_init__(self):
        unrevealed_cards = self.game_state.board.unrevealed_cards
        words = tuple(self.model_format(card.word) for card in unrevealed_cards)
        colors = tuple(card.color for card in unrevealed_cards)
        vectors_list = [v for v in self.model[words]]
        self.board_data = pd.DataFrame(
            data={
                "color": colors,
                "vectors": vectors_list,
            },
            index=words,
        )

    @cached_property
    def team_unrevealed_cards(self) -> Cards:
        return self.game_state.board.unrevealed_cards_for_color(self.team_card_color)

    @cached_property
    def gray_indices(self) -> np.ndarray:
        return self.board_data.color == CardColor.GRAY

    @cached_property
    def opponent_indices(self) -> np.ndarray:
        return self.board_data.color == self.team_card_color.opponent

    @cached_property
    def black_indices(self) -> np.ndarray:
        return self.board_data.color == CardColor.BLACK

    @cached_property
    def board_vectors(self) -> pd.Series:
        return self.board_data["vectors"]

    def word_group_indices(self, word_group: WordGroup) -> np.ndarray:
        return self.board_data.index.isin(word_group)

    def word_group_vectors(self, word_group: WordGroup) -> pd.Series:
        return self.get_vectors(self.word_group_indices(word_group))

    def get_vectors(self, index: np.ndarray) -> pd.Series:
        return self.board_data[index]["vectors"]

    def model_format(self, word: str) -> str:
        return self.model_adapter.to_model_format(word)

    def board_format(self, word: str) -> str:
        return self.model_adapter.to_board_format(word)

    def get_word_frequency(self, word: str) -> float:
        # TODO: len(self.model.key_to_index) is linear, this should not be linear grading.
        return 1 - self.model.key_to_index[word] / len(self.model.key_to_index)

    def generate_proposals(self, max_group_size: int):
        proposals = []
        for group_size in range(max_group_size, 0, -1):
            group_size_proposals = self.create_proposals_for_group_size(group_size=group_size)
            proposals.extend(group_size_proposals)
        return proposals

    def create_proposals_for_group_size(self, group_size: int) -> List[Proposal]:
        log.debug(f"Creating proposals for group size {wrap(group_size)}...")
        proposals = []
        task_manager = AsyncTaskManager()
        for card_group in itertools.combinations(self.team_unrevealed_cards, group_size):
            word_group = tuple(self.model_format(card.word) for card in card_group)
            task_manager.add_task(self.create_proposals_for_word_group, args=(word_group,))
        for result in task_manager:
            proposals.extend(result)
        return proposals

    def create_proposals_for_word_group(self, word_group: WordGroup) -> List[Proposal]:
        # log.debug(f"Creating proposals for group: {word_group}.")
        vectors = self.model[word_group]  # type: ignore
        centroid = np.mean(vectors, axis=0)
        group_indices = self.word_group_indices(word_group)
        group_vectors = self.get_vectors(group_indices)
        centroid_to_group = cosine_distance(centroid, group_vectors)
        max_centroid_to_group = np.max(centroid_to_group)
        if self.thresholds_filter_active and max_centroid_to_group > self.proposals_thresholds.max_distance_group:
            return []
        # distances = cosine_distance(centroid, vectors)
        # group_similarity = np.mean(distances)
        similarities = self.model.most_similar(centroid, topn=self.similarities_top_n)  # type: ignore
        return self.create_proposals_from_similarities(
            word_group=word_group, group_indices=group_indices, similarities=similarities
        )

    def create_proposals_from_similarities(
        self, word_group: WordGroup, group_indices: np.ndarray, similarities: List[Similarity]
    ) -> List[Proposal]:
        proposals = []
        for similarity in similarities:
            proposal = self.proposal_from_similarity(
                word_group=word_group, group_indices=group_indices, similarity=similarity
            )
            if proposal is not None:
                proposals.append(proposal)
        return proposals

    def proposal_from_similarity(
        self, word_group: WordGroup, group_indices: np.ndarray, similarity: Similarity
    ) -> Optional[Proposal]:
        hint, similarity_score = similarity
        # word = format_word(word)
        if self.should_filter_hint(hint=hint, word_group=word_group, filter_expressions=self.game_state.illegal_words):
            return None
        hint_vector = self.model[hint]
        board_distances: np.ndarray = cosine_distance(hint_vector, self.board_vectors)  # type: ignore
        hint_to_group = board_distances[group_indices]
        hint_to_gray = board_distances[self.gray_indices]
        hint_to_opponent = board_distances[self.opponent_indices]
        hint_to_black = board_distances[self.black_indices]
        proposal = Proposal(
            word_group=word_group,
            hint_word=self.board_format(hint),
            hint_word_frequency=self.get_word_frequency(hint),
            distance_group=np.max(hint_to_group),
            distance_gray=np.min(hint_to_gray),
            distance_opponent=np.min(hint_to_opponent),
            distance_black=np.min(hint_to_black),
            board_distances=self._get_board_distances_dict(board_distances),
        )
        if self.gradual_distances_filter_active and not group_is_closest(proposal=proposal):
            return None
        if not self.proposal_satisfy_thresholds(proposal=proposal):
            return None
        proposal.grade = self.proposal_grade_calculator(proposal)
        return proposal

    def _get_board_distances_dict(self, board_distances: np.ndarray) -> Dict[str, float]:
        order = np.argsort(board_distances)
        return {self.board_data.index[i]: board_distances[i] for i in order}

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
        really_good = distances.min >= -0.05 and distances.total >= 0.45
        return pass_thresholds or really_good


class NaiveHinter(Hinter):
    def __init__(
        self,
        name: str,
        model: KeyedVectors = None,
        model_identifier: ModelIdentifier = None,
        proposals_thresholds: ProposalThresholds = None,
        max_group_size: int = 4,
        model_adapter: ModelFormatAdapter = None,
        gradual_distances_filter_active: bool = True,
        proposal_grade_calculator: Callable[[Proposal], float] = default_proposal_grade_calculator,
    ):
        super().__init__(name=name)
        self.model = model
        self.model_identifier = model_identifier
        self.max_group_size = max_group_size
        self.opponent_card_color = None
        self.proposals_thresholds = proposals_thresholds or DEFAULT_THRESHOLDS
        self.model_adapter = model_adapter or DEFAULT_MODEL_ADAPTER
        self.gradual_distances_filter_active = gradual_distances_filter_active
        self.proposal_grade_calculator = proposal_grade_calculator

    def on_game_start(self, language: str, board: Board):
        if self.model is not None:
            return
        if self.model_identifier and self.model_identifier.language == language:
            self.model = load_model(model_identifier=self.model_identifier)
        else:
            self.model = load_language(language=language)
        self.opponent_card_color = self.team_color.opponent.as_card_color  # type: ignore

    @classmethod
    def pick_best_proposal(cls, proposals: List[Proposal]) -> Proposal:
        log.debug(f"Got {len(proposals)} proposals.")
        if len(proposals) == 0:
            raise NoProposalsFound()
        print_top_n = 5
        proposals.sort(key=lambda proposal: -proposal.grade)
        best_ton_n_repr = "\n".join(str(p) for p in proposals[:print_top_n])
        log.info(f"Best {print_top_n} proposals: \n{best_ton_n_repr}")
        best_proposal = proposals[0]
        log.debug("Picked proposal", extra=best_proposal.dict())
        return best_proposal

    def pick_hint(
        self, game_state: HinterGameState, thresholds_filter_active: bool = True, similarities_top_n: int = 10
    ) -> Hint:
        proposal_generator = NaiveProposalsGenerator(
            model=self.model,
            model_adapter=self.model_adapter,
            game_state=game_state,
            team_card_color=self.team_color.as_card_color,  # type: ignore
            proposals_thresholds=self.proposals_thresholds,
            proposal_grade_calculator=self.proposal_grade_calculator,
            thresholds_filter_active=thresholds_filter_active,
            gradual_distances_filter_active=self.gradual_distances_filter_active,
            similarities_top_n=similarities_top_n,
        )
        proposals = proposal_generator.generate_proposals(self.max_group_size)
        try:
            proposal = self.pick_best_proposal(proposals=proposals)
            word_group_board_format = tuple(self.model_adapter.to_board_format(word) for word in proposal.word_group)
            return Hint(word=proposal.hint_word, card_amount=proposal.card_count, for_words=word_group_board_format)
        except NoProposalsFound:
            log.debug("No legal proposals found.")
            if not thresholds_filter_active and similarities_top_n >= 20:
                random_word = uuid4().hex[:4]
                return Hint(word=random_word, card_amount=1)
            log.info("Trying without thresholds filtering.")
            return self.pick_hint(game_state=game_state, thresholds_filter_active=False, similarities_top_n=50)


def _format_dict_floats(d: dict):
    for k, v in d.items():
        if isinstance(v, dict):
            _format_dict_floats(v)
        if isinstance(v, float):
            d[k] = round(v, 3)

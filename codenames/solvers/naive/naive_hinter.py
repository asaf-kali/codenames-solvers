import itertools
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from codenames.game.base import TeamColor, HinterGameState, Hint, Board, CardColor
from codenames.game.player import Hinter
from codenames.solvers.sna_solvers.sna_hinter import cosine_distance  # type: ignore
from codenames.solvers.utils.model_loader import load_language
from codenames.solvers.utils.models import Similarity
from codenames.utils import wrap

log = logging.getLogger(__name__)

SIMILARITY_LOWER_BOUNDARY = 0.5

WordGroup = Tuple[str, ...]


def should_filter_word(word: str, filter_expressions: Iterable[str]) -> bool:
    # if "_" in word:
    #     return True
    # if word in BANNED_WORDS:
    #     return True
    for bad_word in filter_expressions:
        if word in bad_word or bad_word in word:
            return True
    return False


def format_word(word: str) -> str:
    return word.strip().lower()


class NoProposalsFound(Exception):
    pass


@dataclass
class ProposalThresholds:
    max_distance_group: float
    min_distance_gray: float
    min_distance_opponent: float
    min_distance_black: float


# max_distance_group:     high = more results
# min_distance_gray:      high = less results
# min_distance_opponent:  high = less results
# min_distance_black:     high = less results
DEFAULT_THRESHOLDS = ProposalThresholds(0.50, 0.55, 0.60, 0.65)


@dataclass
class Proposal:
    word_group: WordGroup
    hint_word: str
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


def calculate_proposal_grade(proposal: Proposal) -> float:
    """
    High grade is good.
    """
    return (
        1.4 * len(proposal.word_group)
        - 3.0 * proposal.distance_group
        + 2.0 * proposal.distance_gray
        + 3.0 * proposal.distance_opponent
        + 4.0 * proposal.distance_black
    )


@dataclass
class NaiveProposalsGenerator:
    model: KeyedVectors
    game_state: HinterGameState
    proposals_thresholds: ProposalThresholds
    team_card_color: CardColor
    thresholds_filter_active: bool = True

    def __post_init__(self):
        unrevealed_cards = self.game_state.board.unrevealed_cards
        words = tuple(format_word(card.word) for card in unrevealed_cards)
        colors = tuple(card.color for card in unrevealed_cards)
        vectors_lists_list: List[List[float]] = self.model[words].tolist()  # type: ignore
        vectors_list = [np.array(v) for v in vectors_lists_list]
        self.board_data = pd.DataFrame(
            data={
                "color": colors,
                "vector": vectors_list,
            },
            index=words,
        )
        self.illegal_words = {*self.game_state.board.all_words, *self.game_state.given_hint_words}

    def should_filter_proposal(self, proposal: Proposal) -> bool:
        if not self.thresholds_filter_active:
            return False
        return (
            proposal.distance_group > self.proposals_thresholds.max_distance_group
            or proposal.distance_gray < self.proposals_thresholds.min_distance_gray
            or proposal.distance_opponent < self.proposals_thresholds.min_distance_opponent
            or proposal.distance_black < self.proposals_thresholds.min_distance_black
        )

    def proposal_from_similarity(self, word_group: WordGroup, similarity: Similarity) -> Optional[Proposal]:
        word, similarity_score = similarity
        if should_filter_word(word=word, filter_expressions=self.illegal_words):
            return None
        word_vector = self.model[word]
        word_to_group = cosine_distance(word_vector, self.board_data[self.board_data.index.isin(word_group)]["vector"])
        word_to_gray = cosine_distance(word_vector, self.board_data[self.board_data.color == CardColor.GRAY]["vector"])
        word_to_opponent = cosine_distance(
            word_vector, self.board_data[self.board_data.color == self.team_card_color.opponent]["vector"]
        )
        word_to_black = cosine_distance(
            word_vector, self.board_data[self.board_data.color == CardColor.BLACK]["vector"]
        )
        proposal = Proposal(
            word_group=word_group,
            hint_word=word,
            distance_group=np.max(word_to_group),
            distance_gray=np.min(word_to_gray),
            distance_opponent=np.min(word_to_opponent),
            distance_black=np.min(word_to_black),
        )
        if self.should_filter_proposal(proposal=proposal):
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
        log.debug(f"Creating proposals for group: {word_group}.")
        vectors = self.model[word_group]  # type: ignore
        centroid = np.mean(vectors, axis=0)
        # distances = cosine_distance(centroid, vectors)
        # group_similarity = np.mean(distances)
        similarities = self.model.most_similar(centroid)  # type: ignore
        return self.create_proposals_from_similarities(word_group=word_group, similarities=similarities)

    def create_proposals_for_group_size(self, group_size: int) -> List[Proposal]:
        log.info(f"Creating proposals for group size {wrap(group_size)}...")
        team_cards = self.game_state.board.cards_for_color(self.team_card_color)
        unrevealed_cards = tuple(card for card in team_cards if not card.revealed)
        proposals = []
        for card_group in itertools.combinations(unrevealed_cards, group_size):
            word_group = tuple(card.word for card in card_group)
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
        team_color: TeamColor,
        proposals_thresholds: ProposalThresholds = DEFAULT_THRESHOLDS,
        max_group_size: int = 3,
    ):
        super().__init__(name=name, team_color=team_color)
        self.model: KeyedVectors = None  # type: ignore
        self.opponent_card_color = self.team_color.opponent.as_card_color
        self.max_group_size = max_group_size
        self.proposals_thresholds = proposals_thresholds

    def notify_game_starts(self, language: str, board: Board):
        self.model = load_language(language=language)

    def pick_proposal(self, proposals: List[Proposal]) -> Proposal:
        if len(proposals) == 0:
            raise NoProposalsFound()
        proposals.sort(key=lambda proposal: -proposal.grade)
        best_proposal = proposals[0]
        log.info(f"Picked proposal: {best_proposal}")
        return best_proposal

    def pick_hint(self, game_state: HinterGameState, thresholds_filter_active: bool = True) -> Hint:
        proposal_generator = NaiveProposalsGenerator(
            model=self.model,
            game_state=game_state,
            proposals_thresholds=self.proposals_thresholds,
            team_card_color=self.team_color.as_card_color,
            thresholds_filter_active=thresholds_filter_active,
        )
        proposals = proposal_generator.generate_proposals(self.max_group_size)
        try:
            proposal = self.pick_proposal(proposals=proposals)
            return Hint(proposal.hint_word, proposal.card_count)
        except NoProposalsFound:
            log.info("No legal proposals found.")
            if not thresholds_filter_active:
                return Hint("IDK", 2)
            log.info("Trying without thresholds filtering.")
            return self.pick_hint(game_state=game_state, thresholds_filter_active=False)

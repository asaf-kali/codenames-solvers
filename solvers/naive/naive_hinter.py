import logging
from typing import Callable, List, Optional
from uuid import uuid4

import numpy as np
from codenames.game.board import Board
from codenames.game.move import Hint
from codenames.game.player import Hinter
from codenames.game.state import HinterGameState
from gensim.models import KeyedVectors

from solvers.models import ModelFormatAdapter, ModelIdentifier
from solvers.naive.naive_player import NaivePlayer
from solvers.naive.proposal_generator import (
    DEFAULT_THRESHOLDS,
    NaiveProposalsGenerator,
    Proposal,
    ProposalThresholds,
)

log = logging.getLogger(__name__)


class NoProposalsFound(Exception):
    pass


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


class NaiveHinter(NaivePlayer, Hinter):
    def __init__(
        self,
        name: str,
        model: Optional[KeyedVectors] = None,
        model_identifier: Optional[ModelIdentifier] = None,
        proposals_thresholds: Optional[ProposalThresholds] = None,
        max_group_size: int = 4,
        model_adapter: Optional[ModelFormatAdapter] = None,
        gradual_distances_filter_active: bool = True,
        proposal_grade_calculator: Callable[[Proposal], float] = default_proposal_grade_calculator,
    ):
        super().__init__(name=name, model=model, model_identifier=model_identifier, model_adapter=model_adapter)
        self.max_group_size = max_group_size
        self.opponent_card_color = None
        self.proposals_thresholds = proposals_thresholds or DEFAULT_THRESHOLDS
        self.gradual_distances_filter_active = gradual_distances_filter_active
        self.proposal_grade_calculator = proposal_grade_calculator

    def on_game_start(self, language: str, board: Board):
        super().on_game_start(language=language, board=board)
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

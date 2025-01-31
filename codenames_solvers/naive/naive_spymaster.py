import logging
from typing import Callable, List, Optional
from uuid import uuid4

import numpy as np
from codenames.classic.color import ClassicColor
from codenames.classic.state import ClassicPlayerState
from codenames.duet.card import DuetColor
from codenames.duet.state import DuetPlayerState
from codenames.generic.card import CardColor
from codenames.generic.move import Clue
from codenames.generic.player import Spymaster
from codenames.generic.state import SpymasterState
from codenames.generic.team import Team
from gensim.models import KeyedVectors

from codenames_solvers.models import ModelFormatAdapter, ModelIdentifier
from codenames_solvers.naive.naive_player import NaivePlayer
from codenames_solvers.naive.proposal_generator import (
    DEFAULT_THRESHOLDS,
    NaiveProposalsGenerator,
    Proposal,
    ProposalColors,
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
        + 1.8 * proposal.clue_word_frequency
        - 3.5 * proposal.distance_group  # High group distance is bad.
        + 1.0 * proposal.distance_neutral
        + 2.0 * proposal.distance_opponent
        + 3.0 * proposal.distance_assassin
    )
    return float(np.nan_to_num(grade, nan=-100))


class NaiveSpymaster[C: CardColor, T: Team, S: SpymasterState](NaivePlayer[C, T], Spymaster[C, T, S]):
    def __init__(
        self,
        name: str,
        team: T,
        model: Optional[KeyedVectors] = None,
        model_identifier: Optional[ModelIdentifier] = None,
        proposals_thresholds: Optional[ProposalThresholds] = None,
        model_adapter: Optional[ModelFormatAdapter] = None,
        max_group_size: int = 4,
        gradual_distances_filter_active: bool = True,
        proposal_grade_calculator: Callable[[Proposal], float] = default_proposal_grade_calculator,
    ):
        super().__init__(
            name=name,
            team=team,
            model=model,
            model_identifier=model_identifier,
            model_adapter=model_adapter,
        )
        self.max_group_size = max_group_size
        self.proposals_thresholds = proposals_thresholds or DEFAULT_THRESHOLDS
        self.gradual_distances_filter_active = gradual_distances_filter_active
        self.proposal_grade_calculator = proposal_grade_calculator

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
        log.debug("Picked proposal", extra=best_proposal.model_dump())
        return best_proposal

    def give_clue(self, game_state: S, thresholds_filter_active: bool = True, similarities_top_n: int = 10) -> Clue:
        colors = _get_proposal_colors(game_state)
        proposal_generator = NaiveProposalsGenerator(
            model=self.model,
            model_adapter=self._model_adapter,
            game_state=game_state,
            proposal_colors=colors,
            proposals_thresholds=self.proposals_thresholds,
            proposal_grade_calculator=self.proposal_grade_calculator,
            thresholds_filter_active=thresholds_filter_active,
            gradual_distances_filter_active=self.gradual_distances_filter_active,
            similarities_top_n=similarities_top_n,
        )
        proposals = proposal_generator.generate_proposals(self.max_group_size)
        try:
            proposal = self.pick_best_proposal(proposals=proposals)
            word_group_board_format = tuple(self.board_format(word) for word in proposal.word_group)
            return Clue(word=proposal.clue_word, card_amount=proposal.card_count, for_words=word_group_board_format)
        except NoProposalsFound:
            log.debug("No legal proposals found.")
            if not thresholds_filter_active and similarities_top_n >= 20:
                random_word = uuid4().hex[:4]
                return Clue(word=random_word, card_amount=1)
            log.info("Trying without thresholds filtering.")
            return self.give_clue(game_state=game_state, thresholds_filter_active=False, similarities_top_n=50)


def _get_proposal_colors(state: SpymasterState) -> ProposalColors:
    if isinstance(state, ClassicPlayerState):
        return ProposalColors(
            team=state.current_team.as_card_color,
            opponent=state.current_team.opponent.as_card_color,
            neutral=ClassicColor.NEUTRAL,
            assassin=ClassicColor.ASSASSIN,
        )
    if isinstance(state, DuetPlayerState):
        return ProposalColors(
            team=state.current_team.as_card_color,
            opponent=None,
            neutral=DuetColor.NEUTRAL,
            assassin=DuetColor.ASSASSIN,
        )
    raise NotImplementedError(f"Unsupported state type: {type(state)}")

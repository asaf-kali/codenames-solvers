from typing import List

import pytest

from codenames.game.base import Hint, TeamColor, Card, CardColor
from codenames.game.manager import GameManager, Winner, WinningReason
from codenames.tests.testing_players import PredictedTurn, build_teams, SKIP_GUESS
from codenames.utils import configure_logging

configure_logging()


@pytest.fixture
def cards_10() -> List[Card]:
    return [
        Card("Blue 1", color=CardColor.BLUE),  # 0
        Card("Blue 2", color=CardColor.BLUE),  # 1
        Card("Blue 3", color=CardColor.BLUE),  # 2
        Card("Blue 4", color=CardColor.BLUE),  # 3
        Card("Red 1", color=CardColor.RED),  # 4
        Card("Red 2", color=CardColor.RED),  # 5
        Card("Red 3", color=CardColor.RED),  # 6
        Card("Gray 1", color=CardColor.GRAY),  # 7
        Card("Gray 2", color=CardColor.GRAY),  # 8
        Card("Black", color=CardColor.BLACK),  # 9
    ]


@pytest.fixture
def cards_25() -> List[Card]:
    return [
        Card("Blue 1", color=CardColor.BLUE),
        Card("Blue 2", color=CardColor.BLUE),
        Card("Blue 3", color=CardColor.BLUE),
        Card("Blue 4", color=CardColor.BLUE),
        Card("Blue 5", color=CardColor.BLUE),
        Card("Blue 6", color=CardColor.BLUE),
        Card("Blue 7", color=CardColor.BLUE),
        Card("Blue 8", color=CardColor.BLUE),
        Card("Blue 9", color=CardColor.BLUE),
        Card("Red 1", color=CardColor.RED),
        Card("Red 2", color=CardColor.RED),
        Card("Red 3", color=CardColor.RED),
        Card("Red 4", color=CardColor.RED),
        Card("Red 5", color=CardColor.RED),
        Card("Red 6", color=CardColor.RED),
        Card("Red 7", color=CardColor.RED),
        Card("Red 8", color=CardColor.RED),
        Card("Gray 1", color=CardColor.GRAY),
        Card("Gray 2", color=CardColor.GRAY),
        Card("Gray 3", color=CardColor.GRAY),
        Card("Gray 4", color=CardColor.GRAY),
        Card("Gray 5", color=CardColor.GRAY),
        Card("Gray 6", color=CardColor.GRAY),
        Card("Gray 7", color=CardColor.GRAY),
        Card("Black", color=CardColor.BLACK),
    ]


def test_blue_reveals_all_and_wins(cards_10: List[Card]):
    all_turns = [
        PredictedTurn(hint=Hint("A", 2), guesses=[0, 1, SKIP_GUESS]),
        PredictedTurn(hint=Hint("B", 2), guesses=[4, 5, SKIP_GUESS]),
        PredictedTurn(hint=Hint("C", 2), guesses=[2, 3]),
    ]
    blue_team, red_team = build_teams(all_turns=all_turns)
    manager = GameManager.from_teams(blue_team=blue_team, red_team=red_team)
    manager.run_game(language="english", cards=cards_10)
    assert manager.winner == Winner(TeamColor.BLUE, reason=WinningReason.TARGET_SCORE)


def test_red_reveals_all_and_wins(cards_10: List[Card]):
    all_turns = [
        PredictedTurn(hint=Hint("A", 2), guesses=[0, 1, SKIP_GUESS]),
        PredictedTurn(hint=Hint("B", 2), guesses=[4, 5, SKIP_GUESS]),
        PredictedTurn(hint=Hint("C", 2), guesses=[7]),  # Hits gray
        PredictedTurn(hint=Hint("D", 1), guesses=[6]),
    ]
    blue_team, red_team = build_teams(all_turns=all_turns)
    manager = GameManager.from_teams(blue_team=blue_team, red_team=red_team)
    manager.run_game(language="english", cards=cards_10)
    assert manager.winner == Winner(TeamColor.RED, reason=WinningReason.TARGET_SCORE)


def test_blue_picks_black_and_red_wins(cards_10: List[Card]):
    all_turns = [
        PredictedTurn(hint=Hint("A", 2), guesses=[0, 9]),
    ]
    blue_team, red_team = build_teams(all_turns=all_turns)
    manager = GameManager.from_teams(blue_team=blue_team, red_team=red_team)
    manager.run_game(language="english", cards=cards_10)
    assert manager.winner == Winner(TeamColor.RED, reason=WinningReason.OPPONENT_HITS_BLACK)


def test_blue_picks_red_and_red_wins(cards_10: List[Card]):
    all_turns = [
        PredictedTurn(hint=Hint("A", 2), guesses=[0, 7]),  # Hits gray
        PredictedTurn(hint=Hint("B", 2), guesses=[4, 5, 1]),  # Hits blue
        PredictedTurn(hint=Hint("C", 1), guesses=[2, 6]),  # Hits last red
    ]
    blue_team, red_team = build_teams(all_turns=all_turns)
    manager = GameManager.from_teams(blue_team=blue_team, red_team=red_team)
    manager.run_game(language="english", cards=cards_10)
    assert manager.winner == Winner(TeamColor.RED, reason=WinningReason.TARGET_SCORE)

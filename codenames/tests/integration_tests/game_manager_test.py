from unittest.mock import MagicMock

import pytest

from codenames.game.base import Hint, TeamColor, Card, CardColor, Board, Guess
from codenames.game.manager import GameManager, Winner, WinningReason, PASS_GUESS
from codenames.tests.testing_players import PredictedTurn, build_teams
from codenames.utils import configure_logging

configure_logging()


@pytest.fixture
def board_10() -> Board:
    return Board(
        [
            Card("Card 0", color=CardColor.BLUE),  # 0
            Card("Card 1", color=CardColor.BLUE),  # 1
            Card("Card 2", color=CardColor.BLUE),  # 2
            Card("Card 3", color=CardColor.BLUE),  # 3
            Card("Card 4", color=CardColor.RED),  # 4
            Card("Card 5", color=CardColor.RED),  # 5
            Card("Card 6", color=CardColor.RED),  # 6
            Card("Card 7", color=CardColor.GRAY),  # 7
            Card("Card 8", color=CardColor.GRAY),  # 8
            Card("Card 9", color=CardColor.BLACK),  # 9
        ]
    )


@pytest.fixture
def board_25() -> Board:
    return Board(
        [
            Card("Card 0", color=CardColor.BLUE),
            Card("Card 1", color=CardColor.BLUE),
            Card("Card 2", color=CardColor.BLUE),
            Card("Card 3", color=CardColor.BLUE),
            Card("Card 4", color=CardColor.BLUE),
            Card("Card 5", color=CardColor.BLUE),
            Card("Card 6", color=CardColor.BLUE),
            Card("Card 7", color=CardColor.BLUE),
            Card("Card 8", color=CardColor.BLUE),
            Card("Card 9", color=CardColor.RED),
            Card("Card 10", color=CardColor.RED),
            Card("Card 11", color=CardColor.RED),
            Card("Card 12", color=CardColor.RED),
            Card("Card 13", color=CardColor.RED),
            Card("Card 14", color=CardColor.RED),
            Card("Card 15", color=CardColor.RED),
            Card("Card 16", color=CardColor.RED),
            Card("Card 17", color=CardColor.GRAY),
            Card("Card 18", color=CardColor.GRAY),
            Card("Card 19", color=CardColor.GRAY),
            Card("Card 20", color=CardColor.GRAY),
            Card("Card 21", color=CardColor.GRAY),
            Card("Card 22", color=CardColor.GRAY),
            Card("Card 23", color=CardColor.GRAY),
            Card("Card 24", color=CardColor.BLACK),
        ]
    )


def test_blue_reveals_all_and_wins(board_10: Board):
    all_turns = [
        PredictedTurn(hint=Hint("A", 2), guesses=[0, 1, PASS_GUESS]),
        PredictedTurn(hint=Hint("B", 2), guesses=[4, 5, PASS_GUESS]),
        PredictedTurn(hint=Hint("C", 2), guesses=[2, 3]),
    ]
    blue_team, red_team = build_teams(all_turns=all_turns)
    manager = GameManager.from_teams(blue_team=blue_team, red_team=red_team)
    manager.run_game(language="english", board=board_10)
    assert manager.winner == Winner(TeamColor.BLUE, reason=WinningReason.TARGET_SCORE)


def test_red_reveals_all_and_wins(board_10: Board):
    all_turns = [
        PredictedTurn(hint=Hint("A", 2), guesses=[0, 1, PASS_GUESS]),
        PredictedTurn(hint=Hint("B", 2), guesses=[4, 5, PASS_GUESS]),
        PredictedTurn(hint=Hint("C", 2), guesses=[7]),  # Hits gray
        PredictedTurn(hint=Hint("D", 1), guesses=[6]),
    ]
    blue_team, red_team = build_teams(all_turns=all_turns)
    manager = GameManager.from_teams(blue_team=blue_team, red_team=red_team)
    manager.run_game(language="english", board=board_10)
    assert manager.winner == Winner(TeamColor.RED, reason=WinningReason.TARGET_SCORE)


def test_blue_picks_black_and_red_wins(board_10: Board):
    all_turns = [
        PredictedTurn(hint=Hint("A", 2), guesses=[0, 9]),
    ]
    blue_team, red_team = build_teams(all_turns=all_turns)
    manager = GameManager.from_teams(blue_team=blue_team, red_team=red_team)
    manager.run_game(language="english", board=board_10)
    assert manager.winner == Winner(TeamColor.RED, reason=WinningReason.OPPONENT_HITS_BLACK)


def test_blue_picks_red_and_red_wins(board_10: Board):
    all_turns = [
        PredictedTurn(hint=Hint("A", 2), guesses=[0, 7]),  # Hits gray
        PredictedTurn(hint=Hint("B", 2), guesses=[4, 5, 1]),  # Hits blue
        PredictedTurn(hint=Hint("C", 1), guesses=[2, 6]),  # Hits last red
    ]
    blue_team, red_team = build_teams(all_turns=all_turns)
    manager = GameManager.from_teams(blue_team=blue_team, red_team=red_team)
    manager.run_game(language="english", board=board_10)
    assert manager.winner == Winner(TeamColor.RED, reason=WinningReason.TARGET_SCORE)


def test_hint_subscribers_are_notified(board_10: Board):
    hint_given_subscriber = MagicMock()
    all_turns = [
        PredictedTurn(hint=Hint("A", 2), guesses=[0, 1, PASS_GUESS]),
        PredictedTurn(hint=Hint("B", 2), guesses=[4, 5, PASS_GUESS]),
        PredictedTurn(hint=Hint("C", 2), guesses=[7]),  # Hits gray
        PredictedTurn(hint=Hint("D", 1), guesses=[6]),
    ]
    blue_team, red_team = build_teams(all_turns=all_turns)
    manager = GameManager.from_teams(blue_team=blue_team, red_team=red_team)
    manager.hint_given_subscribers.append(hint_given_subscriber)
    manager.run_game(language="english", board=board_10)

    sent_args = [call[0] for call in hint_given_subscriber.call_args_list]
    assert sent_args == [
        (
            blue_team.hinter,
            Hint(word="A", card_amount=2),
        ),
        (
            red_team.hinter,
            Hint(word="B", card_amount=2),
        ),
        (
            blue_team.hinter,
            Hint(word="C", card_amount=2),
        ),
        (
            red_team.hinter,
            Hint(word="D", card_amount=1),
        ),
    ]


def test_guess_subscribers_are_notified(board_10: Board):
    guess_given_subscriber = MagicMock()
    all_turns = [
        PredictedTurn(hint=Hint("A", 2), guesses=[0, 1, PASS_GUESS]),
        PredictedTurn(hint=Hint("B", 2), guesses=[4, 5, PASS_GUESS]),
        PredictedTurn(hint=Hint("C", 2), guesses=[7]),  # Hits gray
        PredictedTurn(hint=Hint("D", 1), guesses=[6]),
    ]
    blue_team, red_team = build_teams(all_turns=all_turns)
    manager = GameManager.from_teams(blue_team=blue_team, red_team=red_team)
    manager.guess_given_subscribers.append(guess_given_subscriber)
    manager.run_game(language="english", board=board_10)

    sent_args = [call[0] for call in guess_given_subscriber.call_args_list]
    assert sent_args == [
        (
            blue_team.guesser,
            Guess(card_index=0),
        ),
        (
            blue_team.guesser,
            Guess(card_index=1),
        ),
        (
            blue_team.guesser,
            Guess(card_index=PASS_GUESS),
        ),
        (
            red_team.guesser,
            Guess(card_index=4),
        ),
        (
            red_team.guesser,
            Guess(card_index=5),
        ),
        (
            red_team.guesser,
            Guess(card_index=PASS_GUESS),
        ),
        (
            blue_team.guesser,
            Guess(card_index=7),
        ),
        (
            red_team.guesser,
            Guess(card_index=6),
        ),
    ]

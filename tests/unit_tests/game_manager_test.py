from unittest.mock import MagicMock

from codenames.game.base import CardColor, GivenGuess, GivenHint, Guess, Hint, TeamColor
from codenames.game.manager import GameManager, Winner, WinningReason
from tests.constants import board_10
from tests.testing_players import PredictedTurn, TestGuesser, TestHinter, build_teams


def test_game_manager_assigns_team_colors_to_players_on_game_manager_construction():
    blue_hinter = TestHinter([])
    blue_guesser = TestGuesser([])
    red_hinter = TestHinter([])
    red_guesser = TestGuesser([])

    assert blue_hinter.team_color is None
    assert red_hinter.team_color is None
    assert blue_guesser.team_color is None
    assert red_guesser.team_color is None

    GameManager(blue_hinter=blue_hinter, red_hinter=red_hinter, blue_guesser=blue_guesser, red_guesser=red_guesser)

    assert blue_hinter.team_color == TeamColor.BLUE
    assert red_hinter.team_color == TeamColor.RED
    assert blue_guesser.team_color == TeamColor.BLUE
    assert red_guesser.team_color == TeamColor.RED


def test_game_manager_notifies_all_players_on_hint_given():
    all_turns = [
        PredictedTurn(hint=Hint("A", 2), guesses=[0, 1, 2]),
        PredictedTurn(hint=Hint("B", 1), guesses=[4, 9]),
    ]
    blue_team, red_team = build_teams(all_turns=all_turns)
    manager = GameManager.from_teams(blue_team=blue_team, red_team=red_team)
    on_hint_given_mock = MagicMock()
    on_guess_given_mock = MagicMock()
    board = board_10()
    for player in manager.players:
        player.on_hint_given = on_hint_given_mock
        player.on_guess_given = on_guess_given_mock
    manager.run_game(language="english", board=board)

    expected_given_hint_1 = GivenHint("a", 2, TeamColor.BLUE)
    expected_given_hint_2 = GivenHint("b", 1, TeamColor.RED)
    assert on_hint_given_mock.call_count == 2 * 4
    assert on_hint_given_mock.call_args_list[0][1] == {"given_hint": expected_given_hint_1}
    assert on_hint_given_mock.call_args_list[4][1] == {"given_hint": expected_given_hint_2}

    assert on_guess_given_mock.call_count == 5 * 4
    assert on_guess_given_mock.call_args_list[0][1] == {
        "given_guess": GivenGuess(given_hint=expected_given_hint_1, guessed_card=board[0])
    }
    assert on_guess_given_mock.call_args_list[4][1] == {
        "given_guess": GivenGuess(given_hint=expected_given_hint_1, guessed_card=board[1])
    }
    assert on_guess_given_mock.call_args_list[8][1] == {
        "given_guess": GivenGuess(given_hint=expected_given_hint_1, guessed_card=board[2])
    }
    assert on_guess_given_mock.call_args_list[12][1] == {
        "given_guess": GivenGuess(given_hint=expected_given_hint_2, guessed_card=board[4])
    }
    assert on_guess_given_mock.call_args_list[16][1] == {
        "given_guess": GivenGuess(given_hint=expected_given_hint_2, guessed_card=board[9])
    }


def test_game_starts_with_team_with_most_cards():
    blue_team, red_team = build_teams(all_turns=[])
    red_team.hinter.hints = [Hint("A", 2)]
    red_team.guesser.guesses = [Guess(9)]
    manager = GameManager.from_teams(blue_team=blue_team, red_team=red_team)
    board = board_10()
    board._cards[3].color = CardColor.RED
    assert len(board.red_cards) > len(board.blue_cards)
    manager.run_game(language="english", board=board)

    assert manager.winner is not None
    assert manager.winner == Winner(team_color=TeamColor.BLUE, reason=WinningReason.OPPONENT_HIT_BLACK)

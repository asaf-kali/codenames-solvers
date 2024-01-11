from unittest.mock import patch

import pytest
from codenames.boards.builder import generate_board
from codenames.game.board import Board
from codenames.game.color import TeamColor
from codenames.game.move import PASS_GUESS, QUIT_GAME
from codenames.game.player import GamePlayers
from codenames.game.runner import GameRunner
from codenames.game.winner import WinningReason

from solvers.cli import CLIGuesser, CLIHinter


@pytest.fixture
def english_board() -> Board:
    return generate_board(language="english", first_team=TeamColor.BLUE, seed=2)


@patch("builtins.input")
def test_cli_players_game(mock_input, english_board: Board):
    blue_hinter = CLIHinter("Leonardo", team_color=TeamColor.BLUE)
    blue_guesser = CLIGuesser("Bard", team_color=TeamColor.BLUE)
    red_hinter = CLIHinter("Adam", team_color=TeamColor.RED)
    red_guesser = CLIGuesser("Eve", team_color=TeamColor.RED)

    players = GamePlayers.from_collection([blue_hinter, blue_guesser, red_hinter, red_guesser])
    mock_input.side_effect = [
        "YOU_CANNOT_PARSE_THIS",  # Invalid hint, ignore
        "Ice, 4",
        "1",  # Hit
        "smell",  # Hit
        "proof",  # Gray
        "Dudi, 1",
        "YOU_CANNOT_PARSE_THIS",  # Invalid guess, ignore
        "style",  # Hit
        f"{PASS_GUESS}",
        "Fail, 2",
        "god",  # Black
        f"{QUIT_GAME}",
    ]
    runner = GameRunner(players=players, board=english_board)
    winner = runner.run_game()
    assert winner.team_color == TeamColor.RED
    assert winner.reason == WinningReason.OPPONENT_HIT_BLACK

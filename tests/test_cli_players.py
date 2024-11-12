from unittest.mock import patch

import pytest
from codenames.classic.board import ClassicBoard
from codenames.classic.builder import generate_board
from codenames.classic.color import ClassicTeam
from codenames.classic.runner.models import GamePlayers
from codenames.classic.runner.runner import ClassicGameRunner
from codenames.classic.winner import WinningReason
from codenames.generic.move import PASS_GUESS, QUIT_GAME

from solvers.cli import CLIOperative, CLISpymaster


@pytest.fixture
def english_board() -> ClassicBoard:
    return generate_board(language="english", first_team=ClassicTeam.BLUE, seed=2)


@patch("builtins.input")
def test_cli_players_game(mock_input, english_board: ClassicBoard):
    blue_hinter = CLISpymaster("Leonardo", team=ClassicTeam.BLUE)
    blue_guesser = CLIOperative("Bard", team=ClassicTeam.BLUE)
    red_hinter = CLISpymaster("Adam", team=ClassicTeam.RED)
    red_guesser = CLIOperative("Eve", team=ClassicTeam.RED)

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
    runner = ClassicGameRunner(players=players, board=english_board)
    winner = runner.run_game()
    assert winner.team == ClassicTeam.RED
    assert winner.reason == WinningReason.OPPONENT_HIT_ASSASSIN

from unittest.mock import patch

from codenames.classic.board import ClassicBoard
from codenames.classic.color import ClassicTeam
from codenames.classic.runner import ClassicGamePlayers, ClassicGameRunner
from codenames.classic.winner import WinningReason
from codenames.generic.move import PASS_GUESS

from solvers.cli import CLIOperative, CLISpymaster


@patch("builtins.input")
def test_cli_players_game(mock_input, classic_board: ClassicBoard):
    blue_spymaster = CLISpymaster("Leonardo", team=ClassicTeam.BLUE)
    blue_operative = CLIOperative("Bard", team=ClassicTeam.BLUE)
    red_spymaster = CLISpymaster("Adam", team=ClassicTeam.RED)
    red_operative = CLIOperative("Eve", team=ClassicTeam.RED)

    players = ClassicGamePlayers.from_collection(blue_spymaster, blue_operative, red_spymaster, red_operative)
    mock_input.side_effect = [
        "YOU_CANNOT_PARSE_THIS",  # Invalid clue, ignore
        "Ice, 4",  # Clue
        "24",  # Blue, correct
        "gymnast",  # Blue, correct
        "london",  # Neutral
        "Dudi, 1",  # Clue
        "no such card",  # Invalid guess, ignore
        "flood",  # Red, correct
        f"{PASS_GUESS}",  # Pass
        "Fail, 2",  # Clue
        "teenage",  # Assassin
    ]
    runner = ClassicGameRunner(players=players, board=classic_board)
    winner = runner.run_game()
    assert winner.team == ClassicTeam.RED
    assert winner.reason == WinningReason.OPPONENT_HIT_ASSASSIN

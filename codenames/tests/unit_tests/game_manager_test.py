from codenames.game.base import TeamColor
from codenames.game.manager import GameManager
from codenames.tests.testing_players import TestGuesser, TestHinter


def test_game_manager_assigns_team_colors_to_players_on_game_manager_construction():
    blue_hinter = TestHinter([])
    blue_guesser = TestGuesser([])
    red_hinter = TestHinter([])
    red_guesser = TestGuesser([])

    assert blue_hinter.team_color is None
    assert red_hinter.team_color is None
    assert blue_guesser.team_color is None
    assert red_guesser.team_color is None

    GameManager(
        blue_hinter=blue_hinter, red_hinter=red_hinter, blue_guesser=blue_guesser, red_guesser=red_guesser
    )

    assert blue_hinter.team_color == TeamColor.BLUE
    assert red_hinter.team_color == TeamColor.RED
    assert blue_guesser.team_color == TeamColor.BLUE
    assert red_guesser.team_color == TeamColor.RED

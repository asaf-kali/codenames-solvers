from codenames.game.base import CardColor, TeamColor
from codenames.game.player import Player


def test_player_team_card_color():
    p1 = Player("Player 1", team_color=TeamColor.RED)
    p2 = Player("Player 2", team_color=TeamColor.BLUE)
    assert p1.team_card_color == CardColor.RED
    assert p2.team_card_color == CardColor.BLUE

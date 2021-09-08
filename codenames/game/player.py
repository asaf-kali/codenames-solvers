from enum import Enum

from codenames.game.base import TeamColor, Hint, HinterGameState, Guess, GuesserGameState, Board


class PlayerRole(Enum):
    HINTER = "Hinter"
    GUESSER = "Guesser"


class Player:
    name: str
    team_color: TeamColor

    def __init__(self, name: str, team_color: TeamColor):
        self.name = name
        self.team_color = team_color

    def __str__(self):
        return f"{self.name} - {self.team_color.value} {self.role.value}"

    @property
    def role(self) -> PlayerRole:
        raise NotImplementedError()

    def notify_game_starts(self, language: str, board: Board):
        pass


class Hinter(Player):
    @property
    def role(self) -> PlayerRole:
        return PlayerRole.HINTER

    def pick_hint(self, game_state: HinterGameState) -> Hint:
        raise NotImplementedError()


class Guesser(Player):
    @property
    def role(self) -> PlayerRole:
        return PlayerRole.GUESSER

    def guess(self, game_state: GuesserGameState) -> Guess:
        raise NotImplementedError()

from enum import Enum

from codenames.game.base import TeamColor, Hint, GameState, Guess, GivenHint


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


class Hinter(Player):
    @property
    def role(self) -> PlayerRole:
        return PlayerRole.HINTER

    def notify_game_starts(self, language: str, state: GameState):
        pass

    def pick_hint(self, state: GameState) -> Hint:
        raise NotImplementedError()


class Guesser(Player):
    @property
    def role(self) -> PlayerRole:
        return PlayerRole.GUESSER

    def notify_game_starts(self, language: str, state: GameState):
        pass

    def guess(self, state: GameState, given_hint: GivenHint, left_guesses: int) -> Guess:
        raise NotImplementedError()

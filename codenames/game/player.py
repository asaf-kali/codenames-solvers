from dataclasses import dataclass
from enum import Enum

from codenames.game.state import TeamColor, Hint, GameState, Guess, GivenHint


class PlayerRole(Enum):
    HINTER = "Hinter"
    GUESSER = "Guesser"


@dataclass
class Player:
    name: str
    team: TeamColor

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

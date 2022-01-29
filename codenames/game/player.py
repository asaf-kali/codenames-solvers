from enum import Enum
from typing import Optional

from codenames.game.base import (
    Board,
    CardColor,
    Guess,
    GuesserGameState,
    Hint,
    HinterGameState,
    TeamColor,
)


class PlayerRole(Enum):
    HINTER = "Hinter"
    GUESSER = "Guesser"


class Player:
    def __init__(self, name: str):
        self.name: str = name
        self.team_color: Optional[TeamColor] = None

    def __str__(self):
        team = ""
        if self.team_color:
            team = f" {self.team_color}"
        return f"{self.name} -{team} {self.role.value}"

    @property
    def role(self) -> PlayerRole:
        raise NotImplementedError()

    @property
    def is_human(self) -> bool:
        return False

    @property
    def team_card_color(self) -> CardColor:
        if self.team_color is None:
            raise ValueError("Team color not set")
        return self.team_color.as_card_color

    def on_game_start(self, language: str, board: Board):
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

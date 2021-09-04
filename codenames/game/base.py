from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import List, Optional, Tuple

BLACK_AMOUNT = 1


class CardColor(Enum):
    BLUE = "Blue"
    RED = "Red"
    GRAY = "Gray"
    BLACK = "Black"

    @property
    def as_team_color(self) -> "TeamColor":
        if self == CardColor.RED:
            return TeamColor.RED
        if self == CardColor.BLUE:
            return TeamColor.BLUE
        raise ValueError(f"No such team color: {self.value}")


class TeamColor(Enum):
    BLUE = "Blue"
    RED = "Red"

    @property
    def opponent(self) -> "TeamColor":
        return TeamColor.BLUE if self == TeamColor.RED else TeamColor.RED

    @property
    def as_card_color(self) -> CardColor:
        return CardColor.BLUE if self == TeamColor.BLUE else CardColor.RED


@dataclass
class Card:
    word: str
    color: Optional[CardColor]  # None for guessers.
    revealed: bool = False

    def __str__(self) -> str:
        result = self.word
        if self.color:
            result += f" ({self.color.value})"
        result += " V" if self.revealed else " X"
        return result


class Board(List[Card]):
    @property
    def size(self) -> int:
        return len(self)

    @property
    def all_words(self) -> List[str]:
        return [card.word for card in self]

    @cached_property
    def red_cards(self) -> Tuple[Card, ...]:
        return tuple(card for card in self if card.color == CardColor.RED)

    @cached_property
    def blue_cards(self) -> Tuple[Card, ...]:
        return tuple(card for card in self if card.color == CardColor.BLUE)

    @property
    def censured(self) -> "Board":
        return self


@dataclass(frozen=True)
class Hint:
    word: str
    card_amount: int

    def __str__(self) -> str:
        return f"{self.word}, {self.card_amount}"


@dataclass(frozen=True)
class GivenHint:
    word: str
    card_amount: int
    team_color: TeamColor

    def __str__(self) -> str:
        return f"{self.word}, {self.card_amount}"


@dataclass(frozen=True)
class Guess:
    card_index: int


@dataclass(frozen=True)
class GivenGuess:
    given_hint: GivenHint
    guessed_card: Card

    def __str__(self) -> str:
        result = "Correct!" if self.was_correct else "Wrong!"
        return f"'{self.guessed_card}', {result}"

    @cached_property
    def was_correct(self) -> bool:
        return self.team.as_card_color == self.guessed_card.color

    @cached_property
    def team(self) -> TeamColor:
        return self.given_hint.team_color


@dataclass
class HinterGameState:
    board: Board
    given_hints: List[GivenHint]
    given_guesses: List[GivenGuess]


@dataclass
class GuesserGameState:
    board: Board
    given_hints: List[GivenHint]
    given_guesses: List[GivenGuess]
    left_guesses: int
    bonus_given: bool

    @property
    def given_hint(self) -> GivenHint:
        return self.given_hints[-1]

from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import List, Set, Optional

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
        return f"'{self.guessed_card.word}', {result}"

    @cached_property
    def was_correct(self) -> bool:
        return self.team.as_card_color == self.guessed_card.color

    @cached_property
    def team(self) -> TeamColor:
        return self.given_hint.team_color


@dataclass
class GameState:
    cards: List[Card]
    given_hints: List[GivenHint]
    given_guesses: List[GivenGuess]

    @property
    def board_size(self) -> int:
        return len(self.cards)

    @property
    def all_words(self) -> List[str]:
        return [card.word for card in self.cards]

    @property
    def all_colors(self) -> List[CardColor]:
        return [card.color for card in self.cards]

    @property
    def all_reveals(self) -> List[bool]:
        return [card.revealed for card in self.cards]

    @property
    def red_cards(self) -> Set[Card]:
        return {card for card in self.cards if card.color == CardColor.RED}

    @property
    def blue_cards(self) -> Set[Card]:
        return {card for card in self.cards if card.color == CardColor.BLUE}

    @property
    def guesser_censored(self) -> "GameState":
        """
        :return: An identical state, where unrevealed cards colors are removed.
        """
        return self

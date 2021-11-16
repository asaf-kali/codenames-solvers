from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import List, Optional, Tuple

BLACK_AMOUNT = 1

Similarity = Tuple[str, float]
WordGroup = Tuple[str, ...]


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
        raise ValueError(f"No such team color: {self.value}.")

    @property
    def opponent(self) -> "CardColor":
        return self.as_team_color.opponent.as_card_color

    def __lt__(self, other: "CardColor") -> bool:
        return self.value < other.value


class TeamColor(Enum):
    BLUE = "Blue"
    RED = "Red"

    @property
    def opponent(self) -> "TeamColor":
        return TeamColor.BLUE if self == TeamColor.RED else TeamColor.RED

    @property
    def as_card_color(self) -> CardColor:
        return CardColor.BLUE if self == TeamColor.BLUE else CardColor.RED

    def __lt__(self, other: "CardColor") -> bool:
        return self.value < other.value


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

    @property
    def censored(self) -> "Card":
        censored_color = self.color if self.revealed else None
        return Card(word=self.word, color=censored_color, revealed=self.revealed)


class Board(List[Card]):
    @property
    def size(self) -> int:
        return len(self)

    @cached_property
    def all_words(self) -> WordGroup:
        return tuple(card.word.lower() for card in self)

    @property
    def all_colors(self) -> Tuple[CardColor, ...]:
        return tuple(card.color for card in self)  # type: ignore

    @property
    def all_reveals(self) -> Tuple[bool, ...]:
        return tuple(card.revealed for card in self)

    @property
    def unrevealed_cards(self) -> Tuple[Card, ...]:
        return tuple(card for card in self if not card.revealed)

    @cached_property
    def red_cards(self) -> Tuple[Card, ...]:
        return self.cards_for_color(CardColor.RED)

    @cached_property
    def blue_cards(self) -> Tuple[Card, ...]:
        return self.cards_for_color(CardColor.BLUE)

    @property
    def censured(self) -> "Board":
        return Board([card.censored for card in self])

    def cards_for_color(self, card_color: CardColor) -> Tuple[Card, ...]:
        return tuple(card for card in self if card.color == card_color)

    def find_card_index(self, word: str) -> Optional[int]:
        word = word.lower()
        if word not in self.all_words:
            return None
        return self.all_words.index(word)


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

    @property
    def given_hint_words(self) -> WordGroup:
        return tuple(hint.word for hint in self.given_hints)


@dataclass
class GuesserGameState:
    board: Board
    given_hints: List[GivenHint]
    given_guesses: List[GivenGuess]
    left_guesses: int
    bonus_given: bool

    @property
    def current_hint(self) -> GivenHint:
        return self.given_hints[-1]

    @property
    def given_hint_words(self) -> WordGroup:
        return tuple(hint.word for hint in self.given_hints)

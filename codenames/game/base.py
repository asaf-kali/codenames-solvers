from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Iterable, List, Optional, Set, Tuple, Union

from codenames.game.exceptions import CardNotFoundError

BLACK_AMOUNT = 1

Similarity = Tuple[str, float]
WordGroup = Tuple[str, ...]
WordSet = Set[str]


def format_word(word: str) -> str:
    return word.replace("_", " ").strip().lower()


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

    def __hash__(self):
        return hash(f"{self.word}{self.color.value}{self.revealed}")

    @property
    def censored(self) -> "Card":
        censored_color = self.color if self.revealed else None
        return Card(word=self.word, color=censored_color, revealed=self.revealed)

    @cached_property
    def formatted_word(self) -> str:
        return format_word(self.word)


CardSet = Set[Card]


class Board:
    _cards: List[Card]

    def __init__(self, cards: Iterable[Card]):
        self._cards = list(cards)

    def __getitem__(self, item: Union[int, str]) -> Card:
        if isinstance(item, str):
            index = self.find_card_index(item)
            if index is None:
                raise IndexError(f"Item not found: {item}")
            item = index
        if not isinstance(item, int):
            raise ValueError(f"Illegal index type for item: {item}")
        if item < 0 or item >= self.size:
            raise IndexError(f"Index out of bounds: {item}")
        return self._cards[item]

    @property
    def size(self) -> int:
        return len(self._cards)

    @cached_property
    def all_words(self) -> WordGroup:
        return tuple(card.formatted_word for card in self._cards)

    @property
    def all_colors(self) -> Tuple[CardColor, ...]:
        return tuple(card.color for card in self._cards)  # type: ignore

    @property
    def all_reveals(self) -> Tuple[bool, ...]:
        return tuple(card.revealed for card in self._cards)

    @property
    def unrevealed_cards(self) -> CardSet:
        return set(card for card in self._cards if not card.revealed)

    @cached_property
    def red_cards(self) -> CardSet:
        return self.cards_for_color(CardColor.RED)

    @cached_property
    def blue_cards(self) -> CardSet:
        return self.cards_for_color(CardColor.BLUE)

    @property
    def censured(self) -> "Board":
        return Board([card.censored for card in self._cards])

    def cards_for_color(self, card_color: CardColor) -> CardSet:
        return set(card for card in self._cards if card.color == card_color)

    def unrevealed_cards_for_color(self, card_color: CardColor) -> CardSet:
        return set(card for card in self._cards if card.color == card_color and not card.revealed)

    def find_card_index(self, word: str) -> int:
        formatted_word = format_word(word)
        if formatted_word not in self.all_words:
            raise CardNotFoundError(word)
        return self.all_words.index(formatted_word)

    def reset_state(self):
        for card in self._cards:
            card.revealed = False


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

    @cached_property
    def formatted_word(self) -> str:
        return format_word(self.word)


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
    def given_hint_words(self) -> WordSet:
        return set(hint.word for hint in self.given_hints)

    @property
    def illegal_words(self) -> WordSet:
        return {*self.board.all_words, *self.given_hint_words}


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

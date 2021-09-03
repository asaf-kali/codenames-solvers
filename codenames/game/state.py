from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import List, Set, Optional

BLACK_AMOUNT = 1


class TeamColor(Enum):
    RED = "Red"
    BLUE = "Blue"

    @staticmethod
    def opponent(team_color: "TeamColor") -> "TeamColor":
        return TeamColor.RED if team_color == TeamColor.BLUE else TeamColor.BLUE


class CardColor(Enum):
    RED = "Red"
    BLUE = "Blue"
    GRAY = "Gray"
    BLACK = "Black"


@dataclass
class Card:
    word: str
    color: Optional[CardColor]  # None for guessers.
    revealed: bool = False


@dataclass(frozen=True)
class Hint:
    word: str
    card_amount: int


@dataclass(frozen=True)
class GivenHint(Hint):
    team: TeamColor


@dataclass(frozen=True)
class Guess:
    card_index: int


@dataclass(frozen=True)
class GivenGuess:
    given_hint: GivenHint
    guessed_card: Card

    @cached_property
    def was_correct(self) -> bool:
        return self.guessed_card.color.value == self.team.value

    @cached_property
    def team(self) -> TeamColor:
        return self.given_hint.team


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

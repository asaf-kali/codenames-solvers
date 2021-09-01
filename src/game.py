import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Iterable, Set

BLACK_AMOUNT = 1


class CardColor(Enum):
    RED = 0
    BLUE = 1
    WHITE = 2
    BLACK = 3


class Team(Enum):
    RED = 0
    BLUE = 1


@dataclass
class Card:
    word: str
    color: CardColor
    revealed: bool = False


@dataclass
class HintWord:
    word: str
    team: Team


def _extract_random_subset(complete_set: set, subset_size: int) -> Tuple[set, set]:
    subset = set(random.sample(complete_set, subset_size))
    reduced_set = complete_set.difference(subset)
    return reduced_set, subset


def words_to_cards(words: Iterable[str]) -> List[Card]:
    words_set = set(words)
    board_size = len(words_set)
    red_amount = board_size // 3
    blue_amount = red_amount + 1
    white_amount = board_size - red_amount - blue_amount - BLACK_AMOUNT

    words_set, red_words = _extract_random_subset(words_set, red_amount)
    words_set, blue_words = _extract_random_subset(words_set, blue_amount)
    words_set, white_words = _extract_random_subset(words_set, white_amount)
    words_set, black_words = _extract_random_subset(words_set, BLACK_AMOUNT)
    assert len(words_set) == 0

    red_cards = [Card(word, CardColor.RED) for word in red_words]
    blue_cards = [Card(word, CardColor.BLUE) for word in blue_words]
    white_cards = [Card(word, CardColor.WHITE) for word in white_words]
    black_cards = [Card(word, CardColor.BLACK) for word in black_words]

    all_cards = red_cards + blue_cards + white_cards + black_cards
    random.shuffle(all_cards)
    return all_cards


class Game:
    def __init__(self, words: List[str], language: str = "english"):
        self.language = language
        self.hinted_words: List[HintWord] = []
        self.cards: List[Card] = words_to_cards(words)

    @property
    def board_size(self) -> int:
        return len(self.cards)

    @property
    def words(self) -> List[str]:
        return [card.word for card in self.cards]

    @property
    def red_cards(self) -> Set[Card]:
        return {card for card in self.cards if card.color == CardColor.RED}

    @property
    def blue_cards(self) -> Set[Card]:
        return {card for card in self.cards if card.color == CardColor.BLUE}

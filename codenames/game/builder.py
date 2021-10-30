import random
from typing import Iterable, Tuple

from codenames.game.base import BLACK_AMOUNT, Board, Card, CardColor, HinterGameState


def _extract_random_subset(complete_set: set, subset_size: int) -> Tuple[set, set]:
    subset = set(random.sample(complete_set, subset_size))
    reduced_set = complete_set.difference(subset)
    return reduced_set, subset


def words_to_random_board(words: Iterable[str], seed: int = None) -> Board:
    random.seed(a=seed)

    words_set = set(words)
    board_size = len(words_set)
    red_amount = board_size // 3
    blue_amount = red_amount + 1
    gray_amount = board_size - red_amount - blue_amount - BLACK_AMOUNT

    words_set, red_words = _extract_random_subset(words_set, red_amount)
    words_set, blue_words = _extract_random_subset(words_set, blue_amount)
    words_set, gray_words = _extract_random_subset(words_set, gray_amount)
    words_set, black_words = _extract_random_subset(words_set, BLACK_AMOUNT)
    assert len(words_set) == 0

    red_cards = [Card(word, CardColor.RED) for word in red_words]
    blue_cards = [Card(word, CardColor.BLUE) for word in blue_words]
    gray_cards = [Card(word, CardColor.GRAY) for word in gray_words]
    black_cards = [Card(word, CardColor.BLACK) for word in black_words]

    all_cards = red_cards + blue_cards + gray_cards + black_cards
    random.shuffle(all_cards)
    return Board(all_cards)


def build_simple_state(words: Iterable[str]) -> HinterGameState:
    board = words_to_random_board(words)
    return HinterGameState(board=board, given_hints=[], given_guesses=[])

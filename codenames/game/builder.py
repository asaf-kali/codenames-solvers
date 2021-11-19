import random
from typing import Iterable, Tuple, Sequence

from codenames.game.base import BLACK_AMOUNT, Board, Card, CardColor, HinterGameState


def _extract_random_subset(elements: Sequence, subset_size: int) -> Tuple[Sequence, Sequence]:
    sample = random.sample(elements, k=subset_size)
    remaining = tuple(e for e in elements if e not in sample)
    return remaining, sample


def words_to_random_board(words: Iterable[str], seed: int = None) -> Board:
    random.seed(seed)

    words_list = tuple(words)
    board_size = len(words_list)
    red_amount = board_size // 3
    blue_amount = red_amount + 1
    gray_amount = board_size - red_amount - blue_amount - BLACK_AMOUNT

    words_list, red_words = _extract_random_subset(words_list, red_amount)
    words_list, blue_words = _extract_random_subset(words_list, blue_amount)
    words_list, gray_words = _extract_random_subset(words_list, gray_amount)
    words_list, black_words = _extract_random_subset(words_list, BLACK_AMOUNT)
    assert len(words_list) == 0

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

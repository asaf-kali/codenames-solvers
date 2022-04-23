import random
from typing import Iterable, Sequence, Tuple

from codenames.game.base import Board, Card, CardColor

BLACK_AMOUNT = 1


def _extract_random_subset(elements: Sequence, subset_size: int) -> Tuple[tuple, tuple]:
    sample = tuple(random.sample(elements, k=subset_size))
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

    red_cards = [Card(word=word, color=CardColor.RED) for word in red_words]
    blue_cards = [Card(word=word, color=CardColor.BLUE) for word in blue_words]
    gray_cards = [Card(word=word, color=CardColor.GRAY) for word in gray_words]
    black_cards = [Card(word=word, color=CardColor.BLACK) for word in black_words]

    all_cards = red_cards + blue_cards + gray_cards + black_cards
    random.shuffle(all_cards)
    return Board(cards=all_cards)

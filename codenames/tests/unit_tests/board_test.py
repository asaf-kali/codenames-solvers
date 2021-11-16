import pytest

from codenames.game.base import Board
from codenames.game.exceptions import CardNotFoundError
from codenames.tests import constants


@pytest.fixture()
def board_10() -> Board:
    return constants.board_10()


def test_get_board_at_integer_index_returns_card(board_10: Board):
    card = board_10[0]
    assert card.word == "Card 0"


def test_get_board_at_negative_index_raises_error(board_10: Board):
    with pytest.raises(IndexError):
        _ = board_10[-1]


def test_get_board_at_upper_bound_index_raises_error(board_10: Board):
    with pytest.raises(IndexError):
        _ = board_10[10]


def test_get_board_at_existing_word_index_returns_card(board_10: Board):
    card = board_10["Card 0"]
    assert card.word == "Card 0"


def test_get_board_at_non_existing_word_raises_error(board_10: Board):
    with pytest.raises(CardNotFoundError):
        _ = board_10["foo"]


def test_get_board_at_float_index_raises_error(board_10: Board):
    with pytest.raises(ValueError):
        _ = board_10[1.1]  # type: ignore

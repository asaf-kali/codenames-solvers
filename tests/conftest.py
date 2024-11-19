import pytest
from codenames.classic.board import ClassicBoard
from codenames.duet.board import DuetBoard

BOARD_DATA_CLASSIC = {
    "language": "english",
    "cards": [
        {"word": "park", "color": "RED", "revealed": False},
        {"word": "lock", "color": "NEUTRAL", "revealed": False},
        {"word": "drill", "color": "NEUTRAL", "revealed": False},
        {"word": "king", "color": "RED", "revealed": False},
        {"word": "ski", "color": "NEUTRAL", "revealed": False},
        {"word": "queen", "color": "RED", "revealed": False},
        {"word": "ninja", "color": "NEUTRAL", "revealed": False},
        {"word": "violet", "color": "BLUE", "revealed": False},
        {"word": "moon", "color": "BLUE", "revealed": False},
        {"word": "earth", "color": "NEUTRAL", "revealed": False},
        {"word": "parrot", "color": "NEUTRAL", "revealed": False},
        {"word": "london", "color": "NEUTRAL", "revealed": False},
        {"word": "avalanche", "color": "RED", "revealed": False},
        {"word": "flood", "color": "RED", "revealed": False},
        {"word": "teenage", "color": "ASSASSIN", "revealed": False},
        {"word": "spiderman", "color": "RED", "revealed": False},
        {"word": "jupiter", "color": "BLUE", "revealed": False},
        {"word": "high-school", "color": "BLUE", "revealed": False},
        {"word": "egypt", "color": "BLUE", "revealed": False},
        {"word": "newton", "color": "BLUE", "revealed": False},
        {"word": "kiss", "color": "BLUE", "revealed": False},
        {"word": "gymnast", "color": "BLUE", "revealed": False},
        {"word": "mail", "color": "RED", "revealed": False},
        {"word": "paper", "color": "RED", "revealed": False},
        {"word": "tomato", "color": "BLUE", "revealed": False},
    ],
}

BOARD_DATA_DUET = {
    "language": "english",
    "cards": [
        {"word": "queen", "color": "GREEN", "revealed": False},
        {"word": "violet", "color": "GREEN", "revealed": False},
        {"word": "king", "color": "NEUTRAL", "revealed": False},
        {"word": "moon", "color": "GREEN", "revealed": False},
        {"word": "gymnast", "color": "GREEN", "revealed": False},
        {"word": "paper", "color": "GREEN", "revealed": False},
        {"word": "jupiter", "color": "NEUTRAL", "revealed": False},
        {"word": "lock", "color": "NEUTRAL", "revealed": False},
        {"word": "mail", "color": "NEUTRAL", "revealed": False},
        {"word": "london", "color": "NEUTRAL", "revealed": False},
        {"word": "earth", "color": "NEUTRAL", "revealed": False},
        {"word": "avalanche", "color": "NEUTRAL", "revealed": False},
        {"word": "park", "color": "GREEN", "revealed": False},
        {"word": "flood", "color": "GREEN", "revealed": False},
        {"word": "teenage", "color": "GREEN", "revealed": False},
        {"word": "high-school", "color": "GREEN", "revealed": False},
        {"word": "newton", "color": "NEUTRAL", "revealed": False},
        {"word": "ninja", "color": "NEUTRAL", "revealed": False},
        {"word": "drill", "color": "ASSASSIN", "revealed": False},
        {"word": "spiderman", "color": "ASSASSIN", "revealed": False},
        {"word": "parrot", "color": "NEUTRAL", "revealed": False},
        {"word": "tomato", "color": "NEUTRAL", "revealed": False},
        {"word": "kiss", "color": "ASSASSIN", "revealed": False},
        {"word": "egypt", "color": "NEUTRAL", "revealed": False},
        {"word": "ski", "color": "NEUTRAL", "revealed": False},
    ],
}

BOARD_DATA_DUET_SMALL = {
    "language": "english",
    "cards": [
        {"word": "queen", "color": "GREEN", "revealed": False},
        {"word": "violet", "color": "GREEN", "revealed": False},
        {"word": "king", "color": "NEUTRAL", "revealed": False},
        {"word": "moon", "color": "GREEN", "revealed": False},
        {"word": "gymnast", "color": "NEUTRAL", "revealed": False},
        {"word": "spiderman", "color": "ASSASSIN", "revealed": False},
        {"word": "parrot", "color": "NEUTRAL", "revealed": False},
        {"word": "tomato", "color": "NEUTRAL", "revealed": False},
        {"word": "kiss", "color": "ASSASSIN", "revealed": False},
        {"word": "egypt", "color": "GREEN", "revealed": False},
        {"word": "ski", "color": "NEUTRAL", "revealed": False},
        {"word": "avalanche", "color": "GREEN", "revealed": False},
        {"word": "park", "color": "NEUTRAL", "revealed": False},
        {"word": "flood", "color": "NEUTRAL", "revealed": False},
    ],
}

BOARD_DATA_DUET_SMALL_DUAL = {
    "language": "english",
    "cards": [
        {"word": "queen", "color": "NEUTRAL", "revealed": False},
        {"word": "violet", "color": "GREEN", "revealed": False},
        {"word": "king", "color": "GREEN", "revealed": False},
        {"word": "moon", "color": "NEUTRAL", "revealed": False},
        {"word": "gymnast", "color": "NEUTRAL", "revealed": False},
        {"word": "spiderman", "color": "ASSASSIN", "revealed": False},
        {"word": "parrot", "color": "NEUTRAL", "revealed": False},
        {"word": "tomato", "color": "NEUTRAL", "revealed": False},
        {"word": "kiss", "color": "ASSASSIN", "revealed": False},
        {"word": "egypt", "color": "GREEN", "revealed": False},
        {"word": "ski", "color": "NEUTRAL", "revealed": False},
        {"word": "avalanche", "color": "GREEN", "revealed": False},
        {"word": "park", "color": "NEUTRAL", "revealed": False},
        {"word": "flood", "color": "GREEN", "revealed": False},
    ],
}


@pytest.fixture
def classic_board() -> ClassicBoard:
    return ClassicBoard.model_validate(BOARD_DATA_CLASSIC)


@pytest.fixture
def duet_board() -> DuetBoard:
    return DuetBoard.model_validate(BOARD_DATA_DUET)


@pytest.fixture
def duet_board_small() -> DuetBoard:
    return DuetBoard.model_validate(BOARD_DATA_DUET_SMALL)


@pytest.fixture
def duet_board_small_dual() -> DuetBoard:
    return DuetBoard.model_validate(BOARD_DATA_DUET_SMALL_DUAL)

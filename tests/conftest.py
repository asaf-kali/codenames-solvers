import pytest
from codenames.classic.board import ClassicBoard

BOARD_DATA = {
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


@pytest.fixture
def english_board() -> ClassicBoard:
    return ClassicBoard.model_validate(BOARD_DATA)

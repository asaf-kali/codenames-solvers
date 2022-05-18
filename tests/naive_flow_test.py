from unittest import mock

import pandas as pd
import pytest
from codenames.boards import build_board
from codenames.game import Board, GameRunner
from gensim.models import KeyedVectors

from solvers.naive.naive_guesser import NaiveGuesser
from solvers.naive.naive_hinter import NaiveHinter

VECTORS_FILE_NAME = "tests/small_model.csv"
BOARD_WORDS = [
    "cloak",
    "kiss",
    "flood",
    "mail",
    "skates",
    "paper",
    "frog",
    "skyscraper",
    "moon",
    "egypt",
    "teacher",
    "avalanche",
    "newton",
    "violet",
    "drill",
    "fever",
    "ninja",
    "jupiter",
    "ski",
    "attic",
    "beach",
    "lock",
    "earth",
    "park",
    "gymnast",
    "king",
    "queen",
    "teenage",
    "tomato",
    "parrot",
    "london",
    "spiderman",
    "high-school",
]
HINT_WORDS = [
    "classmate",
    "roommate",
    "graduated",
    "prince",
    "letter",
    "bedroom",
    "sonar",
    "scuba",
    "shoot",
    "copy",
    "printed",
    "hammer",
    "dam",
    "milton",
    "bow",
    "orbit",
    "planets",
    "maynard",
    "mars",
    "ghostbusters",
    "canucks",
    "giraffe",
    "snowboard",
]
ALL_WORDS = BOARD_WORDS + HINT_WORDS


@pytest.fixture
def english_board() -> Board:
    return build_board(vocabulary=BOARD_WORDS, seed=2)


def mock_load_word2vec_format(*args, **kwargs):
    vectors = pd.read_csv(VECTORS_FILE_NAME, header=None)
    model = KeyedVectors(vector_size=50)
    model[ALL_WORDS] = vectors.values
    return model


@mock.patch("gensim.models.KeyedVectors.load", new=mock_load_word2vec_format)
def test_complete_naive_flow(english_board: Board):
    blue_hinter = NaiveHinter("Leonardo")
    blue_guesser = NaiveGuesser("Bard")
    red_hinter = NaiveHinter("Adam")
    red_guesser = NaiveGuesser("Eve")

    runner = GameRunner(blue_hinter, red_hinter, blue_guesser, red_guesser)
    runner.run_game(language="english", board=english_board)

    assert runner.state.winner is not None

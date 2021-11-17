from unittest import mock

import pandas as pd
import pytest
from gensim.models import KeyedVectors

from codenames.game.base import Board, TeamColor
from codenames.game.builder import words_to_random_board
from codenames.game.manager import GameManager
from codenames.solvers.naive.naive_guesser import NaiveGuesser
from codenames.solvers.naive.naive_hinter import NaiveHinter
from codenames.utils import configure_logging

configure_logging()
VECTORS_FILE_NAME = "codenames/tests/small_model.csv"
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
EXTRA_WORDS = [
    "skaters",
    "skating",
    "mushroom",
    "lipstick",
    "goat",
    "turtle",
    "royal",
    "cute",
    "crown",
    "preteen",
    "gangster",
    "tank",
    "instructor",
    "links",
    "sites",
    "canucks",
    "shootout",
    "caper",
    "ghostbusters",
    "student",
    "doctor",
    "gardens",
    "riverside",
]
ALL_WORDS = BOARD_WORDS + EXTRA_WORDS


@pytest.fixture
def english_board() -> Board:
    return words_to_random_board(words=BOARD_WORDS, seed=3)


def mock_load_word2vec_format(*args, **kwargs):
    vectors = pd.read_csv(VECTORS_FILE_NAME, header=None)
    model = KeyedVectors(vector_size=50)
    model[ALL_WORDS] = vectors.values
    return model


@mock.patch("gensim.models.KeyedVectors.load_word2vec_format", new=mock_load_word2vec_format)
def test_complete_naive_flow(english_board: Board):
    blue_hinter = NaiveHinter("Leonardo", team_color=TeamColor.BLUE)
    blue_guesser = NaiveGuesser("Bard", team_color=TeamColor.BLUE)
    red_hinter = NaiveHinter("Adam", team_color=TeamColor.RED)
    red_guesser = NaiveGuesser("Eve", team_color=TeamColor.RED)

    game_manager = GameManager(blue_hinter, red_hinter, blue_guesser, red_guesser)
    game_manager.run_game(language="english", board=english_board)

    assert game_manager.winner is not None

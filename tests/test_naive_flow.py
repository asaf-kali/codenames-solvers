from unittest import mock

import pandas as pd
import pytest
from codenames.game.board import Board
from codenames.game.color import TeamColor
from codenames.game.player import GamePlayers
from codenames.game.runner import GameRunner
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
    return Board.from_vocabulary(language="english", first_team=TeamColor.BLUE, vocabulary=BOARD_WORDS, seed=2)


def mock_load_word2vec_format(*args, **kwargs):
    vectors = pd.read_csv(VECTORS_FILE_NAME, header=None)
    model = KeyedVectors(vector_size=50)
    model[ALL_WORDS] = vectors.values
    return model


@pytest.mark.slow
@mock.patch("gensim.models.KeyedVectors.load", new=mock_load_word2vec_format)
def test_complete_naive_flow(english_board: Board):
    blue_hinter = NaiveHinter("Leonardo", team_color=TeamColor.BLUE)
    blue_guesser = NaiveGuesser("Bard", team_color=TeamColor.BLUE)
    red_hinter = NaiveHinter("Adam", team_color=TeamColor.RED)
    red_guesser = NaiveGuesser("Eve", team_color=TeamColor.RED)

    players = GamePlayers.from_collection([blue_hinter, blue_guesser, red_hinter, red_guesser])
    runner = GameRunner(players, board=english_board)
    runner.run_game()

    assert runner.state.winner is not None

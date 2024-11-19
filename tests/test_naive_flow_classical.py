from unittest import mock

import pandas as pd
import pytest
from codenames.classic.board import ClassicBoard
from codenames.classic.color import ClassicTeam
from codenames.classic.runner import ClassicGamePlayers, ClassicGameRunner
from gensim.models import KeyedVectors

from solvers.naive.naive_operative import NaiveOperative
from solvers.naive.naive_spymaster import NaiveSpymaster
from tests.resources.resource_manager import get_resource_path
from tests.resources.words import ALL_WORDS

VECTORS_FILE_NAME = get_resource_path("small_model.csv")


def mock_load_word2vec_format(*args, **kwargs):
    vectors = pd.read_csv(VECTORS_FILE_NAME, header=None)
    model = KeyedVectors(vector_size=50)
    model[ALL_WORDS] = vectors.values
    return model


@pytest.mark.slow
@mock.patch("gensim.models.KeyedVectors.load", new=mock_load_word2vec_format)
def test_complete_naive_flow(english_board: ClassicBoard):
    blue_spymaster = NaiveSpymaster("Leonardo", team=ClassicTeam.BLUE)
    blue_operative = NaiveOperative("Bard", team=ClassicTeam.BLUE)
    red_spymaster = NaiveSpymaster("Adam", team=ClassicTeam.RED)
    red_operative = NaiveOperative("Eve", team=ClassicTeam.RED)

    players = ClassicGamePlayers.from_collection(blue_spymaster, blue_operative, red_spymaster, red_operative)
    runner = ClassicGameRunner(players, board=english_board)
    runner.run_game()

    assert runner.state.winner is not None

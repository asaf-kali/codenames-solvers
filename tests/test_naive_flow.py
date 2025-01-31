from unittest import mock

import pandas as pd
import pytest
from codenames.classic.board import ClassicBoard
from codenames.classic.runner import ClassicGamePlayers, ClassicGameRunner
from codenames.classic.team import ClassicTeam
from codenames.duet.board import DuetBoard
from codenames.duet.runner import DuetGamePlayers, DuetGameRunner
from codenames.duet.score import TARGET_REACHED
from codenames.duet.state import DuetGameState
from codenames.duet.team import DuetTeam
from gensim.models import KeyedVectors

from codenames_solvers.naive.naive_duet import UnifiedDuetPlayer
from codenames_solvers.naive.naive_operative import NaiveOperative
from codenames_solvers.naive.naive_spymaster import NaiveSpymaster
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
def test_complete_naive_flow_classic(classic_board: ClassicBoard):
    blue_spymaster = NaiveSpymaster("Leonardo", team=ClassicTeam.BLUE)
    blue_operative = NaiveOperative("Bard", team=ClassicTeam.BLUE)
    red_spymaster = NaiveSpymaster("Adam", team=ClassicTeam.RED)
    red_operative = NaiveOperative("Eve", team=ClassicTeam.RED)

    players = ClassicGamePlayers.from_collection(blue_spymaster, blue_operative, red_spymaster, red_operative)
    runner = ClassicGameRunner(players, board=classic_board)
    runner.run_game()

    assert runner.state.winner is not None


@pytest.mark.slow
@mock.patch("gensim.models.KeyedVectors.load", new=mock_load_word2vec_format)
def test_complete_naive_flow_duet(duet_board_small: DuetBoard, duet_board_small_dual: DuetBoard):
    spymaster = NaiveSpymaster(name="", team=DuetTeam.MAIN)
    operative = NaiveOperative(name="", team=DuetTeam.MAIN)
    player_a = UnifiedDuetPlayer(name="Alice", spymaster=spymaster, operative=operative)
    player_b = UnifiedDuetPlayer(name="Bob", spymaster=spymaster, operative=operative)

    game_state = DuetGameState.from_boards(board_a=duet_board_small, board_b=duet_board_small_dual)
    players = DuetGamePlayers(player_a=player_a, player_b=player_b)
    runner = DuetGameRunner(players=players, state=game_state)
    runner.run_game()

    assert runner.state.game_result == TARGET_REACHED

import os

from codenames.game.manager import GameManager  # noqa
from codenames.solvers import (  # type: ignore  # noqa
    NaiveGuesser,
    NaiveHinter,
    SnaHinter,
)
from codenames.utils import configure_logging
from language_data.model_loader import IS_STEMMED_ENV_KEY, MODEL_NAME_ENV_KEY  # noqa
from playground.boards.english import *  # noqa
from playground.boards.hebrew import *  # noqa
from playground.model_adapters import HEBREW_SUFFIX_ADAPTER

configure_logging()

# English
# os.environ[MODEL_NAME_ENV_KEY] = "wiki-50"
# os.environ[MODEL_NAME_ENV_KEY] = "google-300"

# Hebrew
# os.environ[MODEL_NAME_ENV_KEY] = "wiki-100"
# os.environ[MODEL_NAME_ENV_KEY] = "skv-ft"
os.environ[MODEL_NAME_ENV_KEY] = "skv-v1"
os.environ[IS_STEMMED_ENV_KEY] = "1"


def run_offline():
    blue_hinter = NaiveHinter("Leonardo")
    blue_guesser = NaiveGuesser("Bard")
    red_hinter = NaiveHinter("Adam")
    red_guesser = NaiveGuesser("Eve")

    game_manager = GameManager(blue_hinter, red_hinter, blue_guesser, red_guesser)

    # game_manager.run_game(language="english", board=ENGLISH_BOARD_1)  # noqa

    for player in game_manager.players:
        player.model_adapter = HEBREW_SUFFIX_ADAPTER
    game_manager.run_game(language="hebrew", board=HEBREW_BOARD_1)  # noqa


run_offline()

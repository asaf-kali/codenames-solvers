import os

from codenames.game.manager import GameManager
from codenames.solvers.naive import NaiveGuesser
from codenames.solvers.sna_solvers import SnaHinter  # type: ignore
from codenames.utils import configure_logging
from language_data.model_loader import IS_STEMMED_ENV_KEY, MODEL_NAME_ENV_KEY
from playground.boards.english import *  # noqa
from playground.boards.hebrew import *  # noqa

configure_logging()

# English
# os.environ[MODEL_NAME_ENV_KEY] = "wiki-50"
# os.environ[MODEL_NAME_ENV_KEY] = "google-300"

# Hebrew
# os.environ[MODEL_NAME_ENV_KEY] = "wiki-100"
os.environ[MODEL_NAME_ENV_KEY] = "skv-ft"
os.environ[MODEL_NAME_ENV_KEY] = "skv-v1"
os.environ[IS_STEMMED_ENV_KEY] = "1"


def run_offline():
    blue_hinter = SnaHinter("Leonardo")
    blue_guesser = NaiveGuesser("Bard")
    red_hinter = SnaHinter("Adam")
    red_guesser = NaiveGuesser("Eve")

    game_manager = GameManager(blue_hinter, red_hinter, blue_guesser, red_guesser)

    # game_manager.run_game(language="english", board=ENGLISH_BOARD_1)  # noqa
    game_manager.run_game(language="hebrew", board=HEBREW_BOARD_1)  # noqa


run_offline()

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
os.environ[MODEL_NAME_ENV_KEY] = "wiki-100"


# os.environ[MODEL_NAME_ENV_KEY] = "skv-ft"
# os.environ[MODEL_NAME_ENV_KEY] = "skv-ft"
# os.environ[IS_STEMMED_ENV_KEY] = "1"


def run_offline():
    adapter = HEBREW_SUFFIX_ADAPTER  # DEFAULT_MODEL_ADAPTER

    blue_hinter = NaiveHinter("Leonardo", model_adapter=adapter)
    blue_guesser = NaiveGuesser("Bard", model_adapter=adapter)
    red_hinter = NaiveHinter("Adam", model_adapter=adapter)
    red_guesser = NaiveGuesser("Eve", model_adapter=adapter)

    game_manager = GameManager(blue_hinter, red_hinter, blue_guesser, red_guesser)

    # game_manager.run_game(language="english", board=ENGLISH_BOARD_1)  # noqa
    game_manager.run_game(language="hebrew", board=HEBREW_BOARD_1)  # noqa


run_offline()

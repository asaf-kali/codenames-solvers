import logging
import os

from codenames.game import DEFAULT_MODEL_ADAPTER, Board, QuitGame  # noqa
from codenames.game.manager import GameManager  # noqa
from codenames.solvers import (  # type: ignore  # noqa
    CliGuesser,
    NaiveGuesser,
    NaiveHinter,
    SnaHinter,
)
from codenames.utils import configure_logging
from codenames.utils.loader.model_loader import (  # noqa
    IS_STEMMED_ENV_KEY,
    MODEL_NAME_ENV_KEY,
    ModelIdentifier,
    load_model_async,
)
from playground.boards.english import *  # noqa
from playground.boards.hebrew import *  # noqa
from playground.model_adapters import HEBREW_SUFFIX_ADAPTER  # noqa
from playground.printer import print_results

configure_logging(level="INFO", mute_solvers=False)
log = logging.getLogger(__name__)

# model_id = ModelIdentifier("english", "wiki-50", False)
# model_id = ModelIdentifier("english", "google-300", False)
# model_id = ModelIdentifier("hebrew", "twitter", False)
# model_id = ModelIdentifier("hebrew", "ft-200", False)
model_id = ModelIdentifier("hebrew", "skv-ft-150", True)
# model_id = ModelIdentifier("hebrew", "skv-cbow-150", True)

os.environ[MODEL_NAME_ENV_KEY] = model_id.model_name
os.environ[IS_STEMMED_ENV_KEY] = "1" if model_id.is_stemmed else ""
adapter = HEBREW_SUFFIX_ADAPTER if model_id.language == "hebrew" and model_id.is_stemmed else DEFAULT_MODEL_ADAPTER


def run_offline(board: Board = HEBREW_BOARD_7):  # noqa
    log.info("Running offline game...")
    game_manager = None
    try:
        blue_hinter = NaiveHinter("Einstein", model_adapter=adapter)
        red_hinter = NaiveHinter("Yoda", model_adapter=adapter)
        blue_guesser = NaiveGuesser("Newton", model_adapter=adapter)
        # blue_guesser = CliGuesser("Newton")
        red_guesser = NaiveGuesser("Anakin", model_adapter=adapter)
        game_manager = GameManager(blue_hinter, red_hinter, blue_guesser, red_guesser)
        game_manager.run_game(language=model_id.language, board=board)  # noqa
    except QuitGame:
        log.info("Game quit")
    except:  # noqa
        log.exception("Error occurred")
    finally:
        print_results(game_manager)  # type: ignore


if __name__ == "__main__":
    run_offline()

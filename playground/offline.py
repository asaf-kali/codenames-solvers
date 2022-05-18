import logging
import random

from codenames.game import GameRunner, QuitGame

from playground.boards.english import *  # noqa
from playground.boards.hebrew import *  # noqa
from playground.printer import print_results
from solvers.cli_players import CliGuesser  # noqa
from solvers.naive import NaiveGuesser, NaiveHinter  # noqa
from solvers.naive.naive_cli_guesser import ModelAwareCliGuesser  # noqa
from solvers.utils import DEFAULT_MODEL_ADAPTER, HEBREW_SUFFIX_ADAPTER
from solvers.utils.loader import (  # noqa
    IS_STEMMED_ENV_KEY,
    MODEL_NAME_ENV_KEY,
    ModelIdentifier,
    load_language_async,
    load_model_async,
)
from utils import configure_logging

random.seed(42)
configure_logging(level="DEBUG", mute_solvers=False)
log = logging.getLogger(__name__)

# model_id = ModelIdentifier("english", "wiki-50", False)
# model_id = ModelIdentifier("english", "google-300", False)
# model_id = ModelIdentifier("hebrew", "twitter", False)
# model_id = ModelIdentifier("hebrew", "ft-200", False)
model_id = ModelIdentifier(language="hebrew", model_name="skv-ft-150", is_stemmed=True)
# model_id = ModelIdentifier("hebrew", "skv-cbow-150", True)

# os.environ[MODEL_NAME_ENV_KEY] = model_id.model_name
# os.environ[IS_STEMMED_ENV_KEY] = "1" if model_id.is_stemmed else ""
adapter = HEBREW_SUFFIX_ADAPTER if model_id.language == "hebrew" and model_id.is_stemmed else DEFAULT_MODEL_ADAPTER
load_model_async(model_id)


def run_offline(board: Board = HEBREW_BOARD_6):  # noqa: F405
    log.info("Running offline game...")
    game_runner = None
    try:
        # blue_hinter = OlympicHinter("Einstein", model_adapter=adapter)
        # red_hinter = OlympicHinter("Yoda", model_adapter=adapter)
        blue_hinter = NaiveHinter("Einstein", model_identifier=model_id, model_adapter=adapter)
        red_hinter = NaiveHinter("Yoda", model_identifier=model_id, model_adapter=adapter)
        blue_guesser = NaiveGuesser(name="Newton", model_identifier=model_id, model_adapter=adapter)
        # red_guesser = NaiveGuesser(name="Anakin", model_identifier=model_id, model_adapter=adapter)
        red_guesser = CliGuesser(name="Anakin")
        game_runner = GameRunner(blue_hinter, red_hinter, blue_guesser, red_guesser)
        game_runner.run_game(language=model_id.language, board=board)  # noqa
    except QuitGame:
        log.info("Game quit")
    except:  # noqa
        log.exception("Error occurred")
    finally:
        print_results(game_runner)  # type: ignore


if __name__ == "__main__":
    run_offline()

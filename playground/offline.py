import logging
import os
import random

from codenames.game.exceptions import QuitGame
from codenames.game.runner import GameRunner

from playground.boards.english import *  # noqa
from playground.boards.hebrew import *  # noqa
from playground.printer import print_results
from solvers.cli import CliGuesser  # noqa
from solvers.models import (  # noqa
    DEFAULT_MODEL_ADAPTER,
    HEBREW_SUFFIX_ADAPTER,
    IS_STEMMED_ENV_KEY,
    MODEL_NAME_ENV_KEY,
    ModelIdentifier,
    load_language_async,
    load_model_async,
)
from solvers.naive import NaiveGuesser, NaiveHinter  # noqa
from solvers.naive.naive_cli_guesser import ModelAwareCliGuesser  # noqa

random.seed(42)
log = logging.getLogger(__name__)

model_id = ModelIdentifier(language="english", model_name="wiki-50", is_stemmed=False)
# model_id = ModelIdentifier("english", "google-300", False)
# model_id = ModelIdentifier("hebrew", "twitter", False)
# model_id = ModelIdentifier("hebrew", "ft-200", False)
# model_id = ModelIdentifier(language="hebrew", model_name="skv-ft-150", is_stemmed=True)
# model_id = ModelIdentifier("hebrew", "skv-cbow-150", True)

# os.environ[MODEL_NAME_ENV_KEY] = model_id.model_name
# os.environ[IS_STEMMED_ENV_KEY] = "1" if model_id.is_stemmed else ""
adapter = HEBREW_SUFFIX_ADAPTER if model_id.language == "hebrew" and model_id.is_stemmed else DEFAULT_MODEL_ADAPTER
load_model_async(model_id)
GPT_API_KEY = os.getenv("OPENAI_API_KEY", "")


def run_offline(board: Board = ENGLISH_BOARDS[2]):  # noqa: F405
    log.info("Running offline game...")
    log.setLevel(logging.INFO)
    game_runner = None
    try:
        # blue_hinter = GPTHinter(name="Yoda", api_key=GPT_API_KEY)
        blue_hinter = NaiveHinter(name="Yoda", model_adapter=adapter, max_group_size=4)
        red_hinter = NaiveHinter(name="Einstein", model_identifier=model_id, model_adapter=adapter, max_group_size=3)
        # red_hinter = GPTHinter(name="Einstein", api_key=GPT_API_KEY)
        # red_hinter = OlympicHinter(name="Yoda", model_adapter=adapter)
        blue_guesser = CliGuesser(name="Anakin")
        # blue_guesser = GPTGuesser(name="Anakin", api_key=GPT_API_KEY)
        # blue_guesser = NaiveGuesser(name="Anakin", model_identifier=model_id, model_adapter=adapter)
        red_guesser = CliGuesser(name="Newton")
        # red_guesser = GPTGuesser(name="Newton", api_key=GPT_API_KEY)
        # red_guesser = NaiveGuesser(name="Newton", model_identifier=model_id, model_adapter=adapter)
        game_runner = GameRunner(blue_hinter, red_hinter, blue_guesser, red_guesser)
        game_runner.run_game(board=board)
    except QuitGame:
        log.info("Game quit")
    except:  # noqa
        log.exception("Error occurred")
    finally:
        print_results(game_runner)  # type: ignore


if __name__ == "__main__":
    run_offline()

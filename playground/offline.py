import logging
import os
import random

from codenames.game.color import ClassicTeam
from codenames.game.exceptions import QuitGame
from codenames.game.player import GamePlayers
from codenames.game.runner import GameRunner

from playground.boards.english import *  # noqa
from playground.boards.hebrew import *  # noqa
from playground.printer import print_results
from solvers.cli import CLIOperative  # noqa
from solvers.models import (  # noqa
    DEFAULT_MODEL_ADAPTER,
    HEBREW_SUFFIX_ADAPTER,
    IS_STEMMED_ENV_KEY,
    MODEL_NAME_ENV_KEY,
    ModelIdentifier,
    load_language_async,
    load_model_async,
)
from solvers.naive import NaiveOperative, NaiveSpymaster  # noqa
from solvers.other.naive_cli_operative import ModelAwareCliOperative  # noqa

random.seed(42)

log = logging.getLogger(__name__)

model_id = ModelIdentifier(language="english", model_name="wiki-50", is_stemmed=False)
# model_id = ModelIdentifier(language="english", model_name="glove-twitter-25", is_stemmed=False)
# model_id = ModelIdentifier(language="english", model_name="google-300", is_stemmed=False)
# model_id = ModelIdentifier(language="hebrew", model_name="twitter", is_stemmed=False)
# model_id = ModelIdentifier(language="hebrew", model_name="wiki-100", is_stemmed=False)
# model_id = ModelIdentifier(language="hebrew", model_name="skv-ft-150", is_stemmed=True)
# model_id = ModelIdentifier(language="hebrew", model_name="skv-cbow-150", is_stemmed=True)

boards = HEBREW_BOARDS if model_id.language == "hebrew" else ENGLISH_BOARDS  # noqa: F405
adapter = HEBREW_SUFFIX_ADAPTER if model_id.language == "hebrew" and model_id.is_stemmed else DEFAULT_MODEL_ADAPTER
load_model_async(model_id)
GPT_API_KEY = os.getenv("OPENAI_API_KEY", "")


def run_offline(board: Board = boards[1]):  # noqa: F405
    log.info("Running offline game...")
    game_runner = None
    try:
        blue_spymaster = NaiveSpymaster(
            name="Yoda",
            team=ClassicTeam.BLUE,
            model_identifier=model_id,
            model_adapter=adapter,
            max_group_size=4,
        )
        # blue_spymaster = GPTSpymaster(name="Yoda", api_key=GPT_API_KEY)
        red_spymaster = NaiveSpymaster(
            name="Einstein",
            team=ClassicTeam.RED,
            model_identifier=model_id,
            model_adapter=adapter,
            max_group_size=3,
        )
        # red_spymaster = GPTSpymaster(name="Einstein", api_key=GPT_API_KEY)
        # red_spymaster = OlympicSpymaster(name="Yoda", model_adapter=adapter)
        blue_operative = CLIOperative(name="Anakin", team=ClassicTeam.BLUE)
        # blue_operative = NaiveOperative(
        #     name="Anakin",
        #     model_identifier=model_id,
        #     model_adapter=adapter,
        #     team=ClassicTeam.BLUE,
        # )
        # blue_operative = GPTOperative(name="Anakin", api_key=GPT_API_KEY)
        red_operative = NaiveOperative(
            name="Newton",
            model_identifier=model_id,
            model_adapter=adapter,
            team=ClassicTeam.RED,
        )
        # red_operative = GPTOperative(name="Newton", api_key=GPT_API_KEY)
        players = GamePlayers.from_collection([blue_spymaster, blue_operative, red_spymaster, red_operative])
        game_runner = GameRunner(players=players, board=board)
        game_runner.run_game()
    except QuitGame:
        log.info("Game quit")
    except:  # noqa
        log.exception("Error occurred")
    finally:
        print_results(game_runner)  # type: ignore


if __name__ == "__main__":
    run_offline()

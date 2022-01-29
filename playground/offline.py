import logging
import os

from codenames.game import DEFAULT_MODEL_ADAPTER, QuitGame  # noqa
from codenames.game.manager import GameManager  # noqa
from codenames.solvers import (  # type: ignore  # noqa
    CliGuesser,
    NaiveGuesser,
    NaiveHinter,
    SnaHinter,
)
from codenames.utils import configure_logging
from language_data.model_loader import (  # noqa
    IS_STEMMED_ENV_KEY,
    MODEL_NAME_ENV_KEY,
    ModelIdentifier,
    load_model_async,
)
from playground.boards.english import *  # noqa
from playground.boards.hebrew import *  # noqa
from playground.model_adapters import HEBREW_SUFFIX_ADAPTER  # noqa
from playground.printer import print_results

configure_logging(mute_solvers=True)
log = logging.getLogger(__name__)

# model_id, adapter = ModelIdentifier("english", "wiki-50", False), DEFAULT_MODEL_ADAPTER
# model_id, adapter = ModelIdentifier("english", "google-300", False), DEFAULT_MODEL_ADAPTER
# model_id, adapter = ModelIdentifier("hebrew", "ft-200", False), DEFAULT_MODEL_ADAPTER
model_id, adapter = ModelIdentifier("hebrew", "skv-ft-150", True), HEBREW_SUFFIX_ADAPTER

os.environ[MODEL_NAME_ENV_KEY] = model_id.model_name
os.environ[IS_STEMMED_ENV_KEY] = "1" if model_id.is_stemmed else ""


def run_offline():
    game_manager = None
    try:
        # proposals_thresholds = ProposalThresholds()
        blue_hinter = NaiveHinter("Leonardo", model_adapter=adapter)
        blue_guesser = NaiveGuesser("Bard", model_adapter=adapter)
        red_hinter = NaiveHinter("Adam", model_adapter=adapter)
        red_guesser = NaiveGuesser("Eve", model_adapter=adapter)
        game_manager = GameManager(blue_hinter, red_hinter, blue_guesser, red_guesser)
        # game_manager.run_game(language="english", board=ENGLISH_BOARD_1)  # noqa
        game_manager.run_game(language="hebrew", board=HEBREW_BOARD_4)  # noqa
    except QuitGame:
        log.info("Game quit")
    except:  # noqa
        log.exception("Error occurred")
    finally:
        print_results(game_manager)


if __name__ == "__main__":
    run_offline()

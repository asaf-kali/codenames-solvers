import logging
import os

from codenames.game import QuitGame
from codenames.online import NamecodingGameRunner, NamecodingLanguage  # noqa

from playground.printer import print_results
from solvers.cli_players import CliGuesser, CliHinter  # noqa
from solvers.models import (  # noqa
    DEFAULT_MODEL_ADAPTER,
    HEBREW_SUFFIX_ADAPTER,
    IS_STEMMED_ENV_KEY,
    MODEL_NAME_ENV_KEY,
    ModelIdentifier,
    load_language_async,
)
from solvers.naive import NaiveGuesser, NaiveHinter  # noqa
from solvers.olympic.olympic_hinter import OlympicHinter  # noqa
from utils import configure_logging

configure_logging(level="INFO", mute_solvers=False, mute_online=True)
log = logging.getLogger(__name__)

# model_id = ModelIdentifier("english", "wiki-50", False)
# model_id = ModelIdentifier("english", "google-300", False)
# model_id = ModelIdentifier("hebrew", "twitter", False)
model_id = ModelIdentifier(language="hebrew", model_name="ft-200", is_stemmed=False)
# model_id = ModelIdentifier("hebrew", "skv-ft-150", True)
# model_id = ModelIdentifier("hebrew", "skv-cbow-150", True)

os.environ[MODEL_NAME_ENV_KEY] = model_id.model_name
os.environ[IS_STEMMED_ENV_KEY] = "1" if model_id.is_stemmed else ""

load_language_async(language=model_id.language)  # type: ignore
namecoding_language = NamecodingLanguage.HEBREW if model_id.language == "hebrew" else NamecodingLanguage.ENGLISH
adapter = HEBREW_SUFFIX_ADAPTER if model_id.language == "hebrew" and model_id.is_stemmed else DEFAULT_MODEL_ADAPTER


def run_online():
    log.info("Running online game...")
    online_manager = None
    try:
        # blue_hinter = OlympicHinter("Einstein", model_adapter=adapter)
        # red_hinter = OlympicHinter("Yoda", model_adapter=adapter)
        blue_hinter = NaiveHinter("Einstein", model_identifier=model_id, model_adapter=adapter)  # noqa
        red_hinter = NaiveHinter("Yoda", model_identifier=model_id, model_adapter=adapter)  # noqa
        blue_guesser = NaiveGuesser(name="Newton", model_identifier=model_id, model_adapter=adapter)  # noqa
        red_guesser = NaiveGuesser(name="Anakin", model_identifier=model_id, model_adapter=adapter)  # noqa
        online_manager = NamecodingGameRunner(blue_hinter, red_hinter, None, None, show_host=False)
        # online_manager = NamecodingGameRunner(blue_hinter, red_hinter, blue_guesser, red_guesser, show_host=False)
        online_manager.auto_start(language=namecoding_language, clock=False)
    except QuitGame:
        log.info("Game quit")
    except:  # noqa
        log.exception("Error occurred")
    finally:
        print_results(online_manager.game_runner)
        online_manager.close()


if __name__ == "__main__":
    run_online()

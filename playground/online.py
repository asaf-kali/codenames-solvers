import logging
import os

from codenames.game import DEFAULT_MODEL_ADAPTER, QuitGame  # noqa
from codenames.online import NamecodingGameManager, NamecodingLanguage  # noqa
from codenames.solvers import (  # type: ignore  # noqa
    CliGuesser,
    CliHinter,
    NaiveGuesser,
    NaiveHinter,
    SnaHinter,
)
from codenames.utils import configure_logging
from language_data.model_loader import (  # noqa
    IS_STEMMED_ENV_KEY,
    MODEL_NAME_ENV_KEY,
    ModelIdentifier,
    load_language_async,
)
from playground.model_adapters import HEBREW_SUFFIX_ADAPTER  # noqa
from playground.printer import print_results

configure_logging(level="INFO", mute_solvers=False, mute_online=True)
log = logging.getLogger(__name__)

# model_id = ModelIdentifier("english", "wiki-50", False)
# model_id = ModelIdentifier("english", "google-300", False)
# model_id = ModelIdentifier("hebrew", "twitter", False)
# model_id = ModelIdentifier("hebrew", "ft-200", False)
model_id = ModelIdentifier("hebrew", "skv-ft-150", True)
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
        blue_hinter = NaiveHinter("Einstein", model_adapter=adapter)
        red_hinter = NaiveHinter("Yoda", model_adapter=adapter)
        # blue_guesser = NaiveGuesser("Newton", model_adapter=adapter)
        red_guesser = NaiveGuesser("Anakin", model_adapter=adapter)
        online_manager = NamecodingGameManager(blue_hinter, red_hinter, None, red_guesser, show_host=False)
        # online_manager = NamecodingGameManager(blue_hinter, red_hinter, blue_guesser, show_host=False)
        online_manager.auto_start(language=namecoding_language, clock=False)
    except QuitGame:
        log.info("Game quit")
    except:  # noqa
        log.exception("Error occurred")
    finally:
        print_results(online_manager.game_manager)
        online_manager.close()


if __name__ == "__main__":
    run_online()

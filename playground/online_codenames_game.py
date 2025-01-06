import logging
import os

from codenames.classic.team import ClassicTeam
from codenames.generic.exceptions import QuitGame
from codenames.online.codenames_game.adapter import CodenamesGameLanguage, GameConfigs
from codenames.online.codenames_game.runner import CodenamesGameRunner

from codenames_solvers.models import (
    DEFAULT_MODEL_ADAPTER,
    HEBREW_SUFFIX_ADAPTER,
    IS_STEMMED_ENV_KEY,
    MODEL_NAME_ENV_KEY,
    ModelIdentifier,
    load_language_async,
)
from codenames_solvers.naive.naive_operative import NaiveOperative
from codenames_solvers.naive.naive_spymaster import NaiveSpymaster
from playground.printer import print_results

# configure_logging(level="DEBUG", mute_solvers=False, mute_online=False)
log = logging.getLogger(__name__)

# model_id = ModelIdentifier(language="english", model_name="wiki-50", is_stemmed=False)
# model_id = ModelIdentifier("english", "google-300", False)
# model_id = ModelIdentifier("hebrew", "twitter", False)
# model_id = ModelIdentifier(language="hebrew", model_name="ft-200", is_stemmed=False)
model_id = ModelIdentifier(language="hebrew", model_name="skv-ft-150", is_stemmed=True)
# model_id = ModelIdentifier("hebrew", "skv-cbow-150", True)

os.environ[MODEL_NAME_ENV_KEY] = model_id.model_name
os.environ[IS_STEMMED_ENV_KEY] = "1" if model_id.is_stemmed else ""

load_language_async(language=model_id.language)  # type: ignore
namecoding_language = CodenamesGameLanguage.HEBREW if model_id.language == "hebrew" else CodenamesGameLanguage.ENGLISH
adapter = HEBREW_SUFFIX_ADAPTER if model_id.language == "hebrew" and model_id.is_stemmed else DEFAULT_MODEL_ADAPTER


def run_online():
    log.info("Running online game...")
    online_manager = runner = None
    try:
        configs = GameConfigs(language=CodenamesGameLanguage.HEBREW)
        # blue_spymaster = GPTSpymaster(name="Einstein", api_key=GPT_API_KEY)
        blue_spymaster = NaiveSpymaster(
            "Einstein", team=ClassicTeam.BLUE, model_identifier=model_id, model_adapter=adapter
        )
        red_spymaster = NaiveSpymaster(
            name="Yoda", team=ClassicTeam.RED, model_identifier=model_id, model_adapter=adapter
        )
        blue_operative = NaiveOperative(
            name="Newton", team=ClassicTeam.BLUE, model_identifier=model_id, model_adapter=adapter
        )
        # red_operative = GPTOperative(name="Anakin", api_key=GPT_API_KEY)
        red_operative = NaiveOperative(
            name="Anakin", team=ClassicTeam.RED, model_identifier=model_id, model_adapter=adapter
        )
        online_manager = CodenamesGameRunner(
            blue_spymaster=blue_spymaster,
            red_spymaster=red_spymaster,
            blue_operative=blue_operative,
            red_operative=red_operative,
            show_host=True,
            game_configs=configs,
        )
        # online_manager = CodenamesGameGameRunner(
        # blue_spymaster, red_spymaster, blue_operative, red_operative, show_host=False
        # )
        runner = online_manager.auto_start()
    except QuitGame:
        log.info("Game quit")
    except:  # noqa
        log.exception("Error occurred")
    finally:
        print_results(runner)
        online_manager.close()


if __name__ == "__main__":
    run_online()

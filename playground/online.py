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

configure_logging(level="DEBUG", mute_solvers=False, mute_online=False)
log = logging.getLogger(__name__)

# model_id, adapter = ModelIdentifier("english", "wiki-50", False), DEFAULT_MODEL_ADAPTER
# model_id, adapter = ModelIdentifier("english", "google-300", False), DEFAULT_MODEL_ADAPTER
# model_id, adapter = ModelIdentifier("hebrew", "ft-200", False), DEFAULT_MODEL_ADAPTER
model_id, adapter = ModelIdentifier("hebrew", "skv-ft-150", True), HEBREW_SUFFIX_ADAPTER

os.environ[MODEL_NAME_ENV_KEY] = model_id.model_name
os.environ[IS_STEMMED_ENV_KEY] = "1" if model_id.is_stemmed else ""

load_language_async(language=model_id.language)  # type: ignore


def online_game():
    online_manager = None
    try:
        blue_hinter = NaiveHinter("Leonardo", model_adapter=adapter)
        red_hinter = NaiveHinter("Adam", model_adapter=adapter)
        blue_guesser = NaiveGuesser("Bard", model_adapter=adapter)
        red_guesser = NaiveGuesser("Eve", model_adapter=adapter)
        online_manager = NamecodingGameManager(blue_hinter, red_hinter, blue_guesser, red_guesser, show_host=False)
        online_manager.auto_start(language=NamecodingLanguage.HEBREW, clock=False)
        # online_manager.auto_start(language=NamecodingLanguage.ENGLISH, clock=False)
    except QuitGame:
        log.info("Game quit")
    except:  # noqa
        log.exception("Error occurred")
    finally:
        print_results(online_manager.game_manager)
        online_manager.close()


# def adapter_playground():
#     red_hinter = Hinter("Alex")
#     blue_guesser = Guesser("Adam")
#     host_client = NamecodingPlayerAdapter(player=red_hinter)
#     joiner_client = NamecodingPlayerAdapter(player=blue_guesser)
#
#     host_client.open().host_game().choose_role().set_clock(False).set_language(NamecodingLanguage.HEBREW).ready()
#     game_id = host_client.get_game_id()
#     joiner_client.open().join_game(game_id).choose_role().set_clock(True).set_language(
#         NamecodingLanguage.HEBREW
#     ).ready()
#
#     log.info("Done")


if __name__ == "__main__":
    online_game()

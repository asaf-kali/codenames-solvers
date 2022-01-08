import logging
import os
from time import sleep

from codenames.game import QuitGame
from codenames.online import NamecodingGameManager, NamecodingLanguage
from codenames.solvers import (  # type: ignore  # noqa
    NaiveGuesser,
    NaiveHinter,
    SnaHinter,
)
from codenames.utils import configure_logging
from language_data.model_loader import (  # noqa
    IS_STEMMED_ENV_KEY,
    MODEL_NAME_ENV_KEY,
    load_language_async,
)
from playground.model_adapters import HEBREW_SUFFIX_ADAPTER

configure_logging()
log = logging.getLogger(__name__)

# os.environ[MODEL_NAME_ENV_KEY] = "google-300"
# os.environ[MODEL_NAME_ENV_KEY] = "wiki-50"
# os.environ[MODEL_NAME_ENV_KEY] = "wiki-100"
os.environ[MODEL_NAME_ENV_KEY] = "skv-ft"
os.environ[IS_STEMMED_ENV_KEY] = "1"
load_language_async(language="hebrew")  # type: ignore


def online_game():
    online_manager = None
    try:
        adapter = HEBREW_SUFFIX_ADAPTER
        # adapter = DEFAULT_MODEL_ADAPTER
        blue_hinter = NaiveHinter("Leonardo", model_adapter=adapter)
        blue_guesser = NaiveGuesser("Bard", model_adapter=adapter)
        red_hinter = NaiveHinter("Adam", model_adapter=adapter)
        red_guesser = NaiveGuesser("Eve", model_adapter=adapter)
        online_manager = NamecodingGameManager(blue_hinter, red_hinter, blue_guesser, red_guesser)
        online_manager.auto_start(language=NamecodingLanguage.HEBREW, clock=False)
        sleep(1)
    except QuitGame:
        online_manager.close()
    except Exception as e:  # noqa
        log.exception("Error occurred")
    finally:
        if online_manager is not None:
            log.info(f"Winner: {online_manager.winner}")
            online_manager.close()
    log.info("Done")


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


online_game()

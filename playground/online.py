import logging
import os
from time import sleep

from codenames.game import QuitGame
from codenames.online import NamecodingGameManager, NamecodingLanguage
from codenames.solvers.naive import NaiveGuesser, NaiveHinter  # type: ignore  # noqa
from codenames.solvers.sna_solvers import SnaHinter  # type: ignore  # noqa
from codenames.utils import configure_logging
from language_data.model_loader import MODEL_NAME_ENV_KEY, load_language_async

configure_logging()
log = logging.getLogger(__name__)

# os.environ[MODEL_NAME_ENV_KEY] = "google-300"
# os.environ[MODEL_NAME_ENV_KEY] = "wiki-50"
os.environ[MODEL_NAME_ENV_KEY] = "wiki-100"
load_language_async(language="hebrew")  # type: ignore


def online_game():
    online_manager = None
    try:
        blue_hinter = NaiveHinter("Leonardo")
        blue_guesser = NaiveGuesser("Bard")
        red_hinter = SnaHinter("Adam")
        red_guesser = NaiveGuesser("Eve")
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

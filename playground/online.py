# %%

import logging
from time import sleep

from codenames.game.base import TeamColor
from codenames.game.manager import QuitGame
from codenames.game.player import Hinter, Guesser
from codenames.online.online_adapter import NamecodingPlayerAdapter, NamecodingLanguage
from codenames.online.online_game_manager import NamecodingGameManager
from codenames.solvers.cli_players import CliGuesser
from codenames.solvers.sna_solvers.sna_hinter import SnaHinter  # type: ignore
from codenames.utils import configure_logging

configure_logging()
log = logging.getLogger(__name__)


def online_game():
    online_manager = None
    try:
        blue_hinter = SnaHinter("Leonardo", team_color=TeamColor.BLUE)
        blue_guesser = CliGuesser("Bard", team_color=TeamColor.BLUE)
        red_hinter = SnaHinter("Adam", team_color=TeamColor.RED)
        red_guesser = CliGuesser("Eve", team_color=TeamColor.RED)
        online_manager = NamecodingGameManager(blue_hinter, red_hinter, blue_guesser, red_guesser)
        online_manager.auto_start(language=NamecodingLanguage.ENGLISH, clock=False)
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


def adapter_playground():
    red_hinter = Hinter("Alex", team_color=TeamColor.RED)
    blue_guesser = Guesser("Adam", team_color=TeamColor.BLUE)
    host_client = NamecodingPlayerAdapter(player=red_hinter)
    joiner_client = NamecodingPlayerAdapter(player=blue_guesser)

    host_client.open().host_game().choose_role().set_clock(False).set_language(NamecodingLanguage.HEBREW).ready()
    game_id = host_client.get_game_id()
    joiner_client.open().join_game(game_id).choose_role().set_clock(True).set_language(
        NamecodingLanguage.ENGLISH
    ).ready()

    log.info("Done")


# %%
online_game()

# %%
adapter_playground()

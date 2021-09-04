import logging
from time import sleep

from codenames.game.base import TeamColor
from codenames.game.player import Hinter, Guesser
from codenames.online.online_adapter import NamecodingPlayerAdapter, NamecodingLanguage
from codenames.online.online_game_manager import NamecodingGameManager
from codenames.utils import configure_logging

configure_logging()
log = logging.getLogger(__name__)


def online_game():
    red_hinter = Hinter("Adam", team_color=TeamColor.RED)
    red_guesser = Guesser("Eve", team_color=TeamColor.RED)
    blue_hinter = Hinter("Leonardo", team_color=TeamColor.BLUE)
    blue_guesser = Guesser("Bard", team_color=TeamColor.BLUE)
    online_manager = NamecodingGameManager(red_hinter, red_guesser, blue_hinter, blue_guesser)  # noqa: F841
    sleep(1)
    sleep(1)
    sleep(1)
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


if __name__ == "__main__":
    online_game()

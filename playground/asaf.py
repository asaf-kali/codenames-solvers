# %%
import logging
import os
from time import sleep

from codenames.game.base import TeamColor
from codenames.game.builder import words_to_random_board
from codenames.game.manager import GameManager, QuitGame
from codenames.online.online_adapter import NamecodingLanguage
from codenames.online.online_game_manager import NamecodingGameManager
from codenames.solvers.naive.naive_guesser import NaiveGuesser
from codenames.solvers.naive.naive_hinter import NaiveHinter
from codenames.solvers.utils.model_loader import MODEL_NAME_ENV_KEY
from codenames.utils import configure_logging

configure_logging()

log = logging.getLogger(__name__)
os.environ[MODEL_NAME_ENV_KEY] = "wiki-50"

words = [
    "cloak",
    "kiss",
    "flood",
    "mail",
    "skates",
    "paper",
    "frog",
    "skyscraper",
    "moon",
    "egypt",
    "teacher",
    "avalanche",
    "newton",
    "violet",
    "drill",
    "fever",
    "ninja",
    "jupiter",
    "ski",
    "attic",
    "beach",
    "lock",
    "earth",
    "park",
    "gymnast",
    "king",
    "queen",
    "teenage",
    "tomato",
    "parrot",
    "london",
    "spiderman",
]

board = words_to_random_board(words=words, seed=3)

blue_hinter = NaiveHinter("Leonardo", team_color=TeamColor.BLUE)
blue_guesser = NaiveGuesser("Bard", team_color=TeamColor.BLUE)
red_hinter = NaiveHinter("Adam", team_color=TeamColor.RED)
red_guesser = NaiveGuesser("Eve", team_color=TeamColor.RED)

# %% Online
online_manager = None
try:
    online_manager = NamecodingGameManager(blue_hinter, red_hinter, blue_guesser, red_guesser, show_host=True)
    online_manager.auto_start(language=NamecodingLanguage.ENGLISH, clock=False)
    sleep(1)
except QuitGame:
    pass
except Exception as e:  # noqa
    log.exception("Error occurred")
finally:
    if online_manager is not None:
        log.info(f"Winner: {online_manager.winner}")
        online_manager.close()
log.info("Done")

# %% Offline
game_manager = GameManager(blue_hinter, red_hinter, blue_guesser, red_guesser)
game_manager.run_game(language="english", board=board)

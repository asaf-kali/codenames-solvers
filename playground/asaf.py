# %% Imports
import logging
import os
from time import sleep

from codenames.game.base import TeamColor
from codenames.game.builder import words_to_random_board
from codenames.game.exceptions import QuitGame
from codenames.game.manager import GameManager
from codenames.online.online_adapter import NamecodingLanguage
from codenames.online.online_game_manager import NamecodingGameManager
from codenames.solvers.cli_players import CliGuesser
from codenames.solvers.naive.naive_hinter import NaiveHinter
from codenames.utils import configure_logging
from language_data.model_loader import MODEL_NAME_ENV_KEY

configure_logging()

log = logging.getLogger(__name__)

# %% English setup

os.environ[MODEL_NAME_ENV_KEY] = "wiki-50"
# os.environ[MODEL_NAME_ENV_KEY] = "google-300"

english_words = [
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

english_board = words_to_random_board(words=english_words, seed=3)

# %% Hebrew setup

# os.environ[MODEL_NAME_ENV_KEY] = "twitter"  # "wiki"

hebrew_words = [
    "מטען",
    "עלילה",
    "ניצחון",
    "כבש",
    "יוגה",
    "צבי",
    "אף",
    "מפגש",
    "דק",
    "פרץ",
    "שלם",
    "אדם",
    "הרמוניה",
    "זכוכית",
    "חשמל",
    "מעטפת",
    "אנרגיה",
    "קברן",
    "נחת",
    "חייזר",
    "שיר",
    "מיליונר",
    "לפיד",
    "יקום",
    "דרור",
]

hebrew_board = words_to_random_board(words=hebrew_words, seed=1)

# %% Players setup

blue_hinter = NaiveHinter("Leonardo", team_color=TeamColor.BLUE)
blue_guesser = CliGuesser("Bard", team_color=TeamColor.BLUE)
red_hinter = NaiveHinter("Adam", team_color=TeamColor.RED)
red_guesser = CliGuesser("Eve", team_color=TeamColor.RED)

# %% Online
online_manager = None
try:
    online_manager = NamecodingGameManager(blue_hinter, red_hinter, blue_guesser, red_guesser)
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

# %% Offline English
game_manager = GameManager(blue_hinter, red_hinter, blue_guesser, red_guesser)
game_manager.run_game(language="english", board=english_board)

# %% Offline Hebrew
# game_manager = GameManager(blue_hinter, red_hinter, blue_guesser, red_guesser)
# game_manager.run_game(language="hebrew", board=hebrew_board)

# type: ignore

# %%
import os
from codenames.solvers.sna_solvers.sna_hinter import SnaHinter  # noqa: E402
from codenames.solvers.naive.naive_guesser import NaiveGuesser  # noqa: E402
from codenames.solvers.naive.naive_hinter import NaiveHinter  # noqa: E402
from codenames.game.base import TeamColor
from codenames.game.builder import words_to_random_board
from codenames.game.manager import GameManager
from language_data.model_loader import MODEL_NAME_ENV_KEY
from codenames.utils import configure_logging
from logging import getLogger

configure_logging()
os.environ[MODEL_NAME_ENV_KEY] = "wiki-50"
getLogger("matplotlib.font_manager").disabled = True


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
board = words_to_random_board(words=words)
# %%
for i in range(10):

    blue_hinter = SnaHinter("Leonardo", team_color=TeamColor.BLUE, debug_mode=False)
    blue_guesser = NaiveGuesser("Bard", team_color=TeamColor.BLUE)
    red_hinter = SnaHinter("Adam", team_color=TeamColor.RED, debug_mode=False)
    red_guesser = NaiveGuesser("Eve", team_color=TeamColor.RED)
    game_manager = GameManager(blue_hinter, red_hinter, blue_guesser, red_guesser)
    game_manager.run_game(language="english", board=board)


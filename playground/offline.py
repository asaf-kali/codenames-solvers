# %%
import os

from codenames.game.base import TeamColor
from codenames.game.builder import words_to_random_board
from codenames.game.manager import GameManager
from codenames.model_loader import MODEL_NAME_ENV_KEY
from codenames.solvers.cli_players import CliGuesser
from codenames.solvers.sna_solvers.sna_hinter import SnaHinter  # type: ignore
from codenames.solvers.sna_solvers.sna_guesser import SnaGuesser
from codenames.utils import configure_logging

# %%
configure_logging()

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

board = words_to_random_board(words=words, seed=1)
blue_hinter = SnaHinter("Leonardo", team_color=TeamColor.BLUE)
blue_guesser = SnaGuesser("Bard", team_color=TeamColor.BLUE)
red_hinter = SnaHinter("Adam", team_color=TeamColor.RED)
red_guesser = SnaGuesser("Eve", team_color=TeamColor.RED)
game_manager = GameManager(blue_hinter, red_hinter, blue_guesser, red_guesser)

# %% Run game
from codenames.solvers.sna_solvers.sna_hinter import SnaHinter  # type: ignore
os.environ[MODEL_NAME_ENV_KEY] = "wiki-50"
game_manager.run_game(language="english", board=board)

# %%
print()

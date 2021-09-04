# type: ignore

# %%

from codenames.model_loader import load_language

w = load_language("english")

# %%
import numpy as np
from codenames.visualizer import pretty_print_similarities

v1 = w.get_vector("park")
v2 = w.get_vector("beach")
a = w.most_similar((v1 / np.linalg.norm(v1) + v2 / np.linalg.norm(v2)), topn=50)
pretty_print_similarities(a)

# %%
from codenames.game.base import TeamColor
from codenames.game.builder import words_to_random_board
from codenames.game.manager import GameManager
from codenames.solvers.sna_solvers.sna_hinter import SnaHinter
from codenames.online.online_game_manager import NamecodingGameManager
from codenames.online.online_adapter import NamecodingLanguage
from codenames.solvers.cli_players import CliGuesser
import logging

from codenames.utils import configure_logging

configure_logging()

log = logging.getLogger(__name__)
blue_hinter = SnaHinter("Leonardo", team_color=TeamColor.BLUE)
blue_guesser = CliGuesser("Bard", team_color=TeamColor.BLUE)
red_hinter = SnaHinter("Adam", team_color=TeamColor.RED)
red_guesser = CliGuesser("Eve", team_color=TeamColor.RED)

# %%
game_manager = NamecodingGameManager(blue_hinter, red_hinter, blue_guesser, red_guesser)
try:
    game_manager.auto_start(language=NamecodingLanguage.ENGLISH, clock=False)
except Exception as e:
    log.exception("Error occurred")
    raise e
finally:
    game_manager.close()
# %%

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
game_manager = GameManager(blue_hinter, red_hinter, blue_guesser, red_guesser)
game_manager.run_game(language="english", board=board)

import os

from codenames.game.builder import words_to_random_board
from codenames.game.manager import GameManager
from codenames.solvers.naive import NaiveGuesser
from codenames.solvers.sna_solvers import SnaHinter  # type: ignore
from codenames.utils import configure_logging
from language_data.model_loader import MODEL_NAME_ENV_KEY, load_language_async

configure_logging()
os.environ[MODEL_NAME_ENV_KEY] = "wiki-50"
load_language_async("english")

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


def run_offline():
    board = words_to_random_board(words=words, seed=3)

    blue_hinter = SnaHinter("Leonardo")
    blue_guesser = NaiveGuesser("Bard")
    red_hinter = SnaHinter("Adam")
    red_guesser = NaiveGuesser("Eve")

    game_manager = GameManager(blue_hinter, red_hinter, blue_guesser, red_guesser)
    game_manager.run_game(language="english", board=board)


run_offline()

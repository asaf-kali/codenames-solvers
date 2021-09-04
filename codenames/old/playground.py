# type: ignore
import numpy as np

from codenames.model_loader import load_language
from codenames.visualizer import pretty_print_similarities

LANGUAGE_FOLDER = "language_data"
ENGLISH_DATA_FILE = "english.bin"

# %%

w = load_language("english")

# %%
v1 = w.get_vector("park")
v2 = w.get_vector("beach")
a = w.most_similar((v1 / np.linalg.norm(v1) + v2 / np.linalg.norm(v2)), topn=50)
pretty_print_similarities(a)

# %%
from codenames.game.base import TeamColor
from codenames.game.builder import build_simple_state
from codenames.solvers.sna_solvers.sna_hinter import SnaHinter

board_words = [
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
]  # ['king', 'queen', 'teenage', 'tomato', 'parrot', 'london', 'spiderman']

game_state = build_simple_state(words=board_words)
hinter = SnaHinter(name='Michael', team_color=TeamColor.RED)
hinter.notify_game_starts(language="english", state=game_state)
print(f"Hinter guessed: {hinter.pick_hint(state=game_state)}")


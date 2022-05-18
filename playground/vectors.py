import numpy as np

from solvers.utils.loader.model_loader import load_language
from solvers.visualizer import pretty_print_similarities

# %%

w = load_language("english", None)

# %%

v1 = w.get_vector("park")
v2 = w.get_vector("beach")
a = w.most_similar((v1 / np.linalg.norm(v1) + v2 / np.linalg.norm(v2)), topn=50)

pretty_print_similarities(a)

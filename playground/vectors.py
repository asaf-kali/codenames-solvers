import numpy as np

from solvers.models import load_language

from .visualizer import pretty_print_similarities

# %%

w = load_language("english", None)

# %%

v1 = w.get_vector("park")
v2 = w.get_vector("beach")
a = w.most_similar((v1 / np.linalg.norm(v1) + v2 / np.linalg.norm(v2)), topn=50)

pretty_print_similarities(a)

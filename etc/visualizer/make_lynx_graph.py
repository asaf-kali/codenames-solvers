# %% Taken from sna_spymaster
from typing import Dict

import community
import networkx as nx
from codenames.boards import build_board

from codenames_solvers.models import load_language
from playground.visualizer import render

model = load_language("english", "google-300")
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
]
board = build_board(words=words)
print("kaki")
board_size = board.size
vis_graph = nx.Graph()
vis_graph.add_nodes_from(board.all_words)
louvain = nx.Graph(vis_graph)
for i in range(board_size):
    v = board.all_words[i]
    for j in range(i + 1, board_size):
        u = board.all_words[j]
        distance = model.similarity(v, u) + 1
        if distance > 1.13:
            vis_graph.add_edge(v, u, weight=distance)
        louvain_weight = distance**10
        louvain.add_edge(v, u, weight=louvain_weight)
#
word_to_group: Dict[str, int] = community.best_partition(louvain)
nx.set_node_attributes(vis_graph, word_to_group, "group")
render(vis_graph)

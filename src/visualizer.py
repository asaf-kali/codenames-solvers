import webbrowser
from typing import List, Tuple

import networkx as nx
from pyvis.network import Network

RENDERED_FILE_NAME = "nx.html"
ORIGINAL_SCRIPT_NAME = "https://cdnjs.cloudflare.com/ajax/libs/vis/4.16.1/vis-network.min.js"
LOCAL_SCRIPT_NAME = "visualizer/vis-network.min.js"


def render(nx_graph: nx.Graph):
    nt = Network("900px", "900px")
    nt.from_nx(nx_graph)
    nt.show_buttons(filter_="physics")
    nt.show(RENDERED_FILE_NAME)
    with open(RENDERED_FILE_NAME) as file:
        content = file.read()
    content = content.replace(ORIGINAL_SCRIPT_NAME, LOCAL_SCRIPT_NAME)
    with open(RENDERED_FILE_NAME, "w") as file:
        file.write(content)
    webbrowser.open(RENDERED_FILE_NAME)


def pretty_print_similarities(results: List[Tuple[str, float]]):
    for result in results:
        word, grade = result
        print(f"{grade:.3f}: {word}")

import community
import networkx as nx

from src.game import Game, Team
from src.model_loader import load_language
from src.solvers.abstract_hinter import AbstractHinter
from src.visualizer import render


class SnaHinter(AbstractHinter):
    def __init__(self, team: Team, game: Game):
        super().__init__(team=team, game=game)
        self.model = load_language(language=self.game.language)
        self.language_length = len(self.model.index_to_key)
        self.game_vectors = self.model[self.game.words]

    def pick_hint(self) -> str:
        board_size = self.game.board_size
        vis_graph = nx.Graph()
        vis_graph.add_nodes_from(self.game.words)
        louvain = nx.Graph(vis_graph)
        for i in range(board_size):
            v = self.game.words[i]
            for j in range(i + 1, board_size):
                u = self.game.words[j]
                distance = self.model.similarity(v, u) + 1
                if distance > 1.1:
                    vis_graph.add_edge(v, u, weight=distance)
                louvain_weight = distance ** 10
                louvain.add_edge(v, u, weight=louvain_weight)

        partition_object = community.best_partition(louvain)
        # values = [partition_object.get(node) for node in louvain.nodes()]
        # print(f"Values are: {values}")

        nx.set_node_attributes(vis_graph, partition_object, "group")
        render(vis_graph)
        return "hi"

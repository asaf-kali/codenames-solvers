# type: ignore
from typing import Dict, List, NamedTuple

import community
import networkx as nx
import numpy as np
import pandas as pd

from codenames.game.player import Hinter
from codenames.game.base import GameState, TeamColor, Hint
from codenames.model_loader import load_language
from codenames.visualizer import render, pretty_print_similarities


class Similarity(NamedTuple):
    word: str
    grade: float


def _invert_dict(original: dict) -> dict:
    inverted = {}
    for new_value, new_key in original.items():
        inverted.setdefault(new_key, [])
        inverted[new_key].append(new_value)
    return inverted


def filter_similarities(similarities: List[Similarity], words_to_filter_out: List[str]) -> List[Similarity]:
    filtered = []
    for similarity in similarities:
        if similarity.word in words_to_filter_out:
            continue
        filtered.append(similarity)
    return filtered


class SnaHinter(Hinter):
    def __init__(self, name: str, team_color: TeamColor):
        super().__init__(name=name, team_color=team_color)
        self.model = None
        self.language_length = None
        self.board_data = None


    def notify_game_starts(self, language: str, state: GameState):
        self.model = load_language(language=language)
        self.language_length = len(self.model.index_to_key)
        self.board_data = pd.DataFrame(data={'color': state.all_colors,
                                             'is_revealed': state.all_reveals,
                                             'vector': self.model[state.all_words],
                                             'cluster': None},
                                       index=state.all_words)

    def get_vector(self, word: str):
        # if word in self.game.words:
        #     word_index = self.game.words.index(word)
        #     return self.game_vectors[word_index]
        return self.model[word]

    def rate_group(self, words: List[str]) -> float:
        pass

    @staticmethod
    def single_gram_schmidt(v, u):
        v = v / np.linalg.norm(v)
        u = u / np.linalg.norm(u)

        projection_norm = u.T @ v

        o = u - projection_norm * v

        normed_o = o / np.linalg.norm(o)
        return v, normed_o

    def step_away(self, step_away_from, starting_point, arc_radians):
        step_away_from, normed_o = self.single_gram_schmidt(step_away_from, starting_point)

        original_phase = starting_point.T @ step_away_from

        rotated = step_away_from * np.cos(original_phase + arc_radians) + normed_o * np.sin(original_phase + arc_radians)

        rotated_original_size = rotated * np.linalg.norm(starting_point)

        return rotated_original_size

    def cluster_mean(self, cluster_idx):
        return np.mean(self.board_data[self.board_data.cluster == cluster_idx].vector)

    def pick_hint(self):
        best_cluster = self.pick_cluster
        cluster_centroid = np.mean(self.board_data[self.board_data.cluster == best_cluster].vector)
        cluster_size = len(self.board_data[self.board_data.cluster == best_cluster])
        centroid = self.optimize_centroid(cluster_centroid)
        hint_word = self.model.most_similar(centroid)
        hint = Hint(hint_word, cluster_size)
        return hint

    def pick_cluster(self):
        self.cluster()
        clusters_values = []
        unique_clusters = pd.unique(self.board_data.cluster)
        for c in unique_clusters:
            cluster_data = self.board_data[self.board_data.cluster == c]
            clusters_values.append(self.evaluate_cluster(c))
        best_cluster_idx = np.argmax(clusters_values)
        best_cluster = unique_clusters[best_cluster_idx]
        return best_cluster

    def optimize_centroid(self, centroid) -> np.array:
        return np.array([0])

    def evaluate_cluster(self, cluster_idx: int):
        cluster_centroid = self.cluster_mean(cluster_idx)
        closest_opponent_card = self.model.most_similar_to_given('king', ['queen', 'prince'])


        # return len(cluster) * (max)
        pass

    def cluster(self):
        board_size = len(self.board_data.index)
        vis_graph = nx.Graph()
        words = self.board_data.index.to_list()
        vis_graph.add_nodes_from(words)
        louvain = nx.Graph(vis_graph)
        for i in range(board_size):
            v = words[i]
            for j in range(i + 1, board_size):
                u = words[j]
                distance = self.model.similarity(v, u) + 1
                if distance > 1.1:
                    vis_graph.add_edge(v, u, weight=distance)
                louvain_weight = distance ** 10
                louvain.add_edge(v, u, weight=louvain_weight)

        word_to_group: Dict[str, int] = community.best_partition(louvain)
        self.board_data.cluster = self.board_data.index.map(word_to_group)

    # board_size = state.board_size
    # vis_graph = nx.Graph()
    # vis_graph.add_nodes_from(state.all_words)
    # louvain = nx.Graph(vis_graph)
    # for i in range(board_size):
    #     v = state.all_words[i]
    #     for j in range(i + 1, board_size):
    #         u = state.all_words[j]
    #         distance = self.model.similarity(v, u) + 1
    #         if distance > 1.1:
    #             vis_graph.add_edge(v, u, weight=distance)
    #         louvain_weight = distance ** 10
    #         louvain.add_edge(v, u, weight=louvain_weight)
    #
    # word_to_group: Dict[str, int] = community.best_partition(louvain)
    # # self.board_data.cluster = self.board_data.index.map(word_to_group)
    #
    # group_to_words = _invert_dict(word_to_group)
    # # group_grades = {}
    # for group, words in group_to_words.items():
    #     vectors = self.model[words]
    #     average = sum(vectors)
    #     similarities = self.model.most_similar(average)
    #     filtered_similarities = filter_similarities(similarities=similarities, words_to_filter_out=words)
    #     print(f"\n\nFor the word group: {words}, got:")
    #     pretty_print_similarities(filtered_similarities)
    # # values = [partition_object.get(node) for node in louvain.nodes()]
    # # print(f"Values are: {values}")
    #
    # nx.set_node_attributes(vis_graph, word_to_group, "group")
    # render(vis_graph)
    # return Hint("hi", 2)

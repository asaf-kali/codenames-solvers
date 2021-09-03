from typing import Dict, List, NamedTuple

import community
import networkx as nx
import numpy as np

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
        self.game_vectors = None

    def notify_game_starts(self, language: str, state: GameState):
        self.model = load_language(language=language)
        self.language_length = len(self.model.index_to_key)
        self.game_vectors = self.model[state.all_words]

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


    def pick_hint(self, state: GameState) -> Hint:
        board_size = state.board_size
        vis_graph = nx.Graph()
        vis_graph.add_nodes_from(state.all_words)
        louvain = nx.Graph(vis_graph)
        for i in range(board_size):
            v = state.all_words[i]
            for j in range(i + 1, board_size):
                u = state.all_words[j]
                distance = self.model.similarity(v, u) + 1
                if distance > 1.1:
                    vis_graph.add_edge(v, u, weight=distance)
                louvain_weight = distance ** 10
                louvain.add_edge(v, u, weight=louvain_weight)

        word_to_group: Dict[str, int] = community.best_partition(louvain)
        group_to_words = _invert_dict(word_to_group)
        # group_grades = {}
        for group, words in group_to_words.items():
            vectors = self.model[words]
            average = sum(vectors)
            similarities = self.model.most_similar(average)
            filtered_similarities = filter_similarities(similarities=similarities, words_to_filter_out=words)
            print(f"\n\nFor the word group: {words}, got:")
            pretty_print_similarities(filtered_similarities)
        # values = [partition_object.get(node) for node in louvain.nodes()]
        # print(f"Values are: {values}")

        nx.set_node_attributes(vis_graph, word_to_group, "group")
        render(vis_graph)
        return Hint("hi", 2)

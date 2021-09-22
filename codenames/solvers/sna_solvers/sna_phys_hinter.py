# type: ignore
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import community
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from codenames.game.base import TeamColor, Hint, Board, HinterGameState, CardColor
from codenames.solvers.utils.algebra import cosine_distance, single_gram_schmidt
from codenames.solvers.utils.model_loader import load_language
from codenames.solvers.sna_solvers.sna_hinter import SnaHinter


log = logging.getLogger(__name__)
SIMILARITY_LOWER_BOUNDARY = 0.25
MIN_BLACK_DISTANCE = 0.25
MIN_SELF_BLACK_DELTA = 0.15
MIN_SELF_OPPONENT_DELTA = 0.1
MIN_SELF_GRAY_DELTA = 0.05
MAX_SELF_DISTANCE = 0.2
OPPONENT_FORCE_CUTOFF = 0.275
OPPONENT_FORCE_FACTOR = 1.6
FRIENDLY_FORCE_CUTOFF = 0.2
FRIENDLY_FORCE_FACTOR = 1
BLACK_FORCE_FACTOR = 2
GRAY_FORCE_FACTOR = 1.2
EPSILON = 0.001

BANNED_WORDS = {"slackerjack"}
Similarity = Tuple[str, float]


def _invert_dict(original: dict) -> dict:
    inverted = {}
    for new_value, new_key in original.items():
        inverted.setdefault(new_key, [])
        inverted[new_key].append(new_value)
    return inverted


def should_filter_word(word: str, filter_expressions: Iterable[str]) -> bool:
    if "_" in word:
        return True
    if word in BANNED_WORDS:
        return True
    for bad_word in filter_expressions:
        if word in bad_word or bad_word in word:
            return True
    return False


def pick_best_similarity(similarities: List[Similarity], words_to_filter_out: Iterable[str]) -> Optional[Similarity]:
    words_to_filter_out = {word.lower() for word in words_to_filter_out}
    filtered_similarities = []
    for similarity in similarities:
        word, grade = similarity
        word = word.lower()
        if should_filter_word(word, words_to_filter_out):
            continue
        filtered_similarities.append(similarity)
    if len(filtered_similarities) == 0:
        return None
    return filtered_similarities[0]


def step_away(starting_point: np.array, step_away_from: np.array, arc_radians: float) -> np.array:
    step_away_from, normed_o = single_gram_schmidt(step_away_from, starting_point)

    original_phase = np.arccos(starting_point.T @ step_away_from)

    rotated = step_away_from * np.cos(original_phase + arc_radians) + normed_o * np.sin(original_phase + arc_radians)

    rotated_original_size = rotated * np.linalg.norm(starting_point)

    return rotated_original_size


def step_towards(starting_point: np.array, step_away_from: np.array, arc_radians: float) -> np.array:
    return step_away(starting_point, step_away_from, -arc_radians)


def sum_forces(starting_point: np.array, nodes) -> np.array:  # : List[Tuple([)np.array, float], ...]
    # Nodes are vector+Force from vector pairs
    total_force = np.zeros(nodes[0][0].shape)
    for node in nodes:
        rotated = step_away(starting_point, node[0], node[1] * EPSILON)
        contribution = rotated - starting_point
        total_force += contribution
    return total_force


def step_from_forces(
    starting_point: np.array, nodes, arc_radians: float
) -> np.array:  #: List[Tuple[np.array, float], ...]
    net_force = sum_forces(starting_point, nodes)
    force_size = np.linalg.norm(net_force)
    direction_vector = starting_point + net_force
    rotated = step_towards(starting_point, direction_vector, force_size * arc_radians / EPSILON)
    return rotated


def friendly_force(d):
    if d > FRIENDLY_FORCE_CUTOFF:
        return 0
    else:
        # Parabola with 0 at d=0, 1 at d=FRIENDLY_FORCE_CUTOFF and else outherwise:
        return FRIENDLY_FORCE_FACTOR * (
            1 - (d / FRIENDLY_FORCE_CUTOFF - 1) ** 2
        )  # FRIENDLY_FORCE_FACTOR * d / FRIENDLY_FORCE_CUTOFF


def repelling_force(d, cutoff_distance, factor):
    if d > OPPONENT_FORCE_CUTOFF:
        return 0
    else:
        a = factor / (factor - 1) * cutoff_distance
        return a / (d + a / factor)


def opponent_force(d):
    return repelling_force(d, OPPONENT_FORCE_CUTOFF, OPPONENT_FORCE_FACTOR)


def gray_force(d):
    return repelling_force(d, OPPONENT_FORCE_CUTOFF, GRAY_FORCE_FACTOR)


def black_force(d):
    return repelling_force(d, OPPONENT_FORCE_CUTOFF, BLACK_FORCE_FACTOR)


def format_word(word: str) -> str:
    return word.replace(" ", "_").replace("-", "_").strip()


@dataclass
class Cluster:
    id: int
    df: pd.DataFrame
    centroid: Optional[np.array] = None
    grade: float = 0

    @property
    def words(self) -> Tuple[str, ...]:
        return tuple(self.df.index)

    @property
    def default_centroid(self) -> np.array:
        non_normalized_v = np.mean(self.df["vector_normed"])
        normalized_v = non_normalized_v / np.linalg.norm(non_normalized_v)
        return normalized_v

    def __gt__(self, cluster_2):
        return self.grade > cluster_2.grade

    def __lt__(self, cluster_2):
        return self.grade < cluster_2.grade


class SnaPhysHinter(SnaHinter):
    def __init__(self, name: str, team_color: TeamColor, debug_mode):
        super().__init__(name=name, team_color=team_color, debug_mode=debug_mode)

    def optimize_cluster(self, cluster: Cluster, method: str = "Geometric") -> Cluster:
        # Initiate the middle of the cluster as the initial cluster's centroid
        initial_centroid = cluster.default_centroid
        # Rank words by their distance to the centroid:
        cluster.df["centroid_distance"] = cluster.df["vector"].apply(lambda v: cosine_distance(v, initial_centroid))
        cluster.df.sort_values("centroid_distance", inplace=True)
        cluster = self.optimize_centroid_phys(cluster)
        return cluster

    def color2force(self, centroid, row):
        d = cosine_distance(centroid, row["vector"])
        if row["color"] == self.team_color.opponent.as_card_color:
            return opponent_force(d)
        elif row["color"] == CardColor.BLACK:
            return black_force(d)
        elif row["color"] == CardColor.GRAY:
            return gray_force(d)
        elif row["color"] == self.team_color.as_card_color:
            return friendly_force(d)
        else:
            raise ValueError(f"color{row['color']} is not a valid color")

    def board_df2nodes(self, centroid: np.array):  # -> List[Tuple[np.array, float], ...]:
        relevant_df = self.board_data[self.board_data["is_revealed"] == False]
        relevant_df["force"] = relevant_df.apply(lambda row: self.color2force(centroid, row), axis=1)
        relevant_df = relevant_df[["vector", "force"]]
        return list(relevant_df.itertuples(index=False, name=None))

    def optimization_break_condition(self, cluster: Cluster) -> bool:
        self.update_distances(cluster.centroid)
        distances2opponent = self.extract_centroid_distances(self.team_color.opponent.as_card_color)
        distances2own = self.extract_centroid_distances(self.team_color.as_card_color)
        distances2own = distances2own[distances2own.index.isin(cluster.df.index.to_list())]
        distance2black = self.extract_centroid_distances(CardColor.BLACK)
        distances2gray = self.extract_centroid_distances(CardColor.GRAY)
        max_distance2own = max(distances2own)
        if (
            (min(distances2opponent) - max_distance2own > MIN_SELF_OPPONENT_DELTA)
            and (distance2black[0] - max_distance2own > MIN_SELF_OPPONENT_DELTA)
            and (min(distances2gray) - max_distance2own > MIN_SELF_GRAY_DELTA)
            and (max_distance2own < MAX_SELF_DISTANCE)
        ):
            return True
        else:
            return False

    def optimize_centroid_phys(self, cluster: Cluster) -> Cluster:
        # temp_board = self.board_data.copy()
        # temp_board['centroid_distance'] = cluster.df['vector'].apply(lambda v: cosine_distance(v, centroid))
        # temp_board['in_current_cluster'] = temp_board.index.map(lambda x: x in cluster.df.index)
        cluster.centroid = cluster.default_centroid
        if self.debug_mode is True:
            self.draw_guesser_view(cluster)
        for i in range(100):
            self.clean_cluster(cluster)
            if self.debug_mode is True:
                self.draw_guesser_view(cluster)
            if self.optimization_break_condition(cluster):
                break
            nodes = self.board_df2nodes(cluster.centroid)
            cluster.centroid = step_from_forces(cluster.centroid, nodes, arc_radians=5e-2)
            if self.debug_mode is True:
                self.draw_guesser_view(cluster)
        return cluster

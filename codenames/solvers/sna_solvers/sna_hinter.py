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
from codenames.game.player import Hinter
from codenames.solvers.utils.algebra import cosine_distance, single_gram_schmidt
from codenames.solvers.utils.model_loader import load_language

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


class SnaHinter(Hinter):
    def __init__(self, name: str, team_color: TeamColor, debug_mode=True):
        super().__init__(name=name, team_color=team_color)
        self.model: Optional[KeyedVectors] = None
        self.language_length: Optional[int] = None
        self.board_data: Optional[pd.DataFrame] = None
        self.graded_clusters: List[Cluster] = []
        self.debug_mode = debug_mode

    def notify_game_starts(self, language: str, board: Board):
        self.model = load_language(language=language)
        self.language_length = len(self.model.index_to_key)
        all_words = [format_word(word) for word in board.all_words]
        vectors_lists_list: List[List[float]] = self.model[all_words].tolist()  # type: ignore
        vectors_list = [np.array(v) for v in vectors_lists_list]
        vectors_list_normed = [v / np.linalg.norm(v) for v in vectors_list]
        self.board_data = pd.DataFrame(
            data={
                "color": board.all_colors,
                "is_revealed": board.all_reveals,
                "vector": vectors_list,
                "vector_normed": vectors_list_normed,
                "cluster": None,
                "distance_to_centroid": None,
            },
            index=board.all_words,
        )

    @property
    def unrevealed_cards(self) -> pd.DataFrame:
        return self.board_data[self.board_data["is_revealed"] == False]

    @property
    def opponent_cards(self) -> pd.DataFrame:
        return self.board_data[self.board_data["color"] == self.team_color.opponent.as_card_color]

    @property
    def gray_cards(self) -> pd.DataFrame:
        return self.board_data[self.board_data["color"] == CardColor.GRAY]

    @property
    def own_cards(self) -> pd.DataFrame:
        return self.board_data[self.board_data["color"] == self.team_color]

    @property
    def black_card(self) -> pd.DataFrame:
        return self.board_data[self.board_data["color"] == CardColor.BLACK]

    @property
    def bad_cards(self) -> pd.DataFrame:
        return self.board_data[
            self.board_data["color"].isin([CardColor.GRAY, CardColor.BLAC, self.team_color.opponent.as_card_color])
        ]

    def pick_hint(self, game_state: HinterGameState) -> Hint:
        # self.board_data.is_revealed = state.board.all_reveals
        self.generate_graded_clusters()
        for cluster in self.graded_clusters:
            cluster_size = len(cluster.df)
            similarities: List[Similarity] = self.model.most_similar(cluster.centroid)
            cluster_words = cluster.df.index.to_list()
            best_similarity = pick_best_similarity(
                similarities=similarities, words_to_filter_out={*cluster_words, *game_state.given_hint_words}
            )
            if best_similarity is None:
                log.info("No legal similarity found")
                continue
            word, grade = best_similarity
            log.info(f"Cluster words: {cluster.words}, best word: ({word}, {grade})")
            self.draw_guesser_view(cluster)
            if grade < SIMILARITY_LOWER_BOUNDARY:
                log.info(f"Grade wasn't good enough (below {SIMILARITY_LOWER_BOUNDARY})")
                continue
            hint = Hint(word, cluster_size)
            return hint
        return Hint("IDK", 2)

    def generate_graded_clusters(self, resolution_parameter=1):
        unrevealed_self_index = (self.board_data.is_revealed == False) & (  # noqa: E712
            self.board_data.color == self.team_color.as_card_color
        )
        unrevealed_cards = self.board_data[unrevealed_self_index]
        self.divide_to_clusters(df=unrevealed_cards, resolution_parameter=resolution_parameter)
        unrevealed_cards = self.board_data[unrevealed_self_index]  # Now updated with cluster columns
        self.graded_clusters = []
        unique_clusters_ids = pd.unique(unrevealed_cards.cluster)
        for cluster_id in unique_clusters_ids:
            df = unrevealed_cards.loc[unrevealed_cards.cluster == cluster_id, :]
            cluster = Cluster(id=cluster_id, df=df.copy(deep=True))
            optimized_cluster = self.optimize_cluster(cluster)

            self.graded_clusters.append(optimized_cluster)
        self.graded_clusters.sort(key=lambda c: -c.grade)

    def optimize_cluster(self, cluster: Cluster, method: str = "Geometric") -> Cluster:
        raise NotImplementedError()

    def extract_centroid_distances(self, color: CardColor):
        relevant_df = self.board_data[self.board_data["is_revealed"] == False]
        if color == CardColor.GRAY:
            color_rows = relevant_df.color == CardColor.GRAY
        elif color == CardColor.BLACK:
            color_rows = relevant_df.color == CardColor.BLACK
        elif color == self.team_color.opponent.as_card_color:
            color_rows = relevant_df.color == self.team_color.opponent.as_card_color
        elif color == self.team_color.as_card_color:
            color_rows = relevant_df.color == self.team_color.as_card_color
        elif color == CardColor.BAD:
            color_rows = relevant_df.color.isin(
                [CardColor.GRAY, CardColor.BLAC, self.team_color.opponent.as_card_color]
            )
        else:
            raise ValueError(f"No such color as {color}")
        return relevant_df.loc[color_rows, "distance_to_centroid"]


    def update_distances(self, centroid):
        self.board_data["distance_to_centroid"] = self.board_data["vector"].apply(
            lambda v: cosine_distance(v, centroid)
        )

    def clean_cluster(self, cluster: Cluster):
        cluster.df["centroid_distance"] = cluster.df["vector"].apply(lambda v: cosine_distance(v, cluster.centroid))
        max_distance = max(cluster.df["centroid_distance"])
        central_word = (cluster.df["centroid_distance"] < MAX_SELF_DISTANCE) | (cluster.df["centroid_distance"] != max_distance)
        cluster.df = cluster.df[central_word]
        cluster.centroid = cluster.default_centroid

    # flake8: noqa: F841
    def grade_cluster(self, cluster: Cluster) -> float:
        distances = cosine_distance(cluster.centroid, cluster.df.vector)
        centroid_to_black = cosine_distance(
            cluster.centroid, self.board_data[self.board_data.color == CardColor.Black]["vector"]
        )
        centroid_to_gray = np.min(
            cosine_distance(cluster.centroid, self.board_data[self.board_data.color == CardColor.Gray]["vector"])
        )
        centroid_to_opponent = centroid_to_gray = np.min(
            cosine_distance(
                cluster.centroid,
                self.board_data[self.board_data.color == self.team_color.opponent.as_card_color]["vector"],
            )
        )
        return np.mean(distances)  # type: ignore

    def draw_guesser_view(self, cluster: Cluster):
        self.update_distances(cluster.centroid)
        temp_df = self.unrevealed_cards.sort_values("distance_to_centroid")
        colors = temp_df["color"].apply(lambda x: x.value.lower())
        temp_df["is_in_cluster"] = temp_df.index.isin(cluster.df.index)
        edge_color = temp_df["is_in_cluster"].apply(lambda x: 'yellow' if x else 'black')
        line_width = temp_df["is_in_cluster"].apply(lambda x: 3 if x else 1)
        plt.bar(
            x=temp_df.index,
            height=temp_df["distance_to_centroid"],
            color=colors,
            edgecolor=edge_color,
            linewidth=line_width
        )
        ax = plt.gca()
        plt.setp(ax.get_xticklabels(), rotation="vertical")
        similarities: List[Similarity] = self.model.most_similar(cluster.centroid)
        cluster_words = cluster.df.index.to_list()
        best_similarity = pick_best_similarity(
            similarities=similarities, words_to_filter_out={*cluster_words}
        )
        plt.title(best_similarity)
        plt.show()

    def divide_to_clusters(self, df: pd.DataFrame, resolution_parameter=1):
        board_size = len(df)
        vis_graph = nx.Graph()
        words = df.index.to_list()
        vis_graph.add_nodes_from(words)
        louvain = nx.Graph(vis_graph)
        for i in range(board_size):
            v = format_word(words[i])
            for j in range(i + 1, board_size):
                u = format_word(words[j])
                distance = self.model.similarity(v, u) + 1
                if distance > 1.1:
                    vis_graph.add_edge(v, u, weight=distance)
                louvain_weight = distance ** (15 * resolution_parameter)
                louvain.add_edge(v, u, weight=louvain_weight)

        word_to_group: Dict[str, int] = community.best_partition(louvain)
        self.board_data.cluster = self.board_data.index.map(word_to_group)

        # board_size = state.board_size#
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

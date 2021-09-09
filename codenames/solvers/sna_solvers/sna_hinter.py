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
OPPONENT_FORCE_CUTOFF = 0.2
OPPONENT_FORCE_FACTOR = 10
FRIENDLY_FORCE_CUTOFF = 0.2
FRIENDLY_FORCE_FACTOR = 1
BLACK_FORCE_FACTOR = 2

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

    original_phase = starting_point.T @ step_away_from

    rotated = step_away_from * np.cos(original_phase + arc_radians) + normed_o * np.sin(original_phase + arc_radians)

    rotated_original_size = rotated * np.linalg.norm(starting_point)

    return rotated_original_size


def step_towards(starting_point: np.array, step_away_from: np.array, arc_radians: float) -> np.array:
    return step_away(starting_point, step_away_from, -arc_radians)


def sum_forces(starting_point: np.array, nodes) -> np.array:  # : List[Tuple([)np.array, float], ...]
    # Nodes are vector+Force from vector pairs
    total_force = np.zeros(nodes[0][0].shape)
    epsilon = 0.00001
    for node in nodes:
        rotated = step_away(starting_point, node[0], node[1] * epsilon)
        contribution = rotated - starting_point
        total_force += contribution
    return total_force / epsilon


def step_from_forces(
    starting_point: np.array, nodes, arc_radians: float
) -> np.array:  #: List[Tuple[np.array, float], ...]
    net_force = sum_forces(starting_point, nodes)
    direction_vector = starting_point + net_force
    force_size = np.linalg.norm(net_force)
    rotated = step_towards(starting_point, direction_vector, force_size * arc_radians)
    return rotated


def opponent_force(vec, opponent_vec):
    d = cosine_distance(opponent_vec, vec)
    if d > OPPONENT_FORCE_CUTOFF:
        return 0
    else:
        # return force of OPPONENT_FORCE_FACTOR at distance=0, force of 1 at distance = OPPONENT_FORCE_CUTOFF and 0 else
        a = OPPONENT_FORCE_FACTOR / (OPPONENT_FORCE_FACTOR - 1) * OPPONENT_FORCE_CUTOFF
        return a / (d + a / OPPONENT_FORCE_FACTOR)


def friendly_force(vec, friend_vec):
    d = cosine_distance(vec, friend_vec)
    if d > FRIENDLY_FORCE_CUTOFF:
        return 0
    else:
        # Parabola with 0 at d=0, 1 at d=FRIENDLY_FORCE_CUTOFF and else outherwise:
        return FRIENDLY_FORCE_FACTOR * (
            1 - (d / FRIENDLY_FORCE_CUTOFF - 1) ** 2
        )  # FRIENDLY_FORCE_FACTOR * d / FRIENDLY_FORCE_CUTOFF


def gray_force(vec, gray_vec):
    return opponent_force(vec, gray_vec) * FRIENDLY_FORCE_FACTOR


def black_force(vec, gray_vec):
    return opponent_force(vec, gray_vec) * BLACK_FORCE_FACTOR


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
    def __init__(self, name: str, team_color: TeamColor):
        super().__init__(name=name, team_color=team_color)
        self.model: Optional[KeyedVectors] = None
        self.language_length: Optional[int] = None
        self.board_data: Optional[pd.DataFrame] = None
        self.graded_clusters: List[Cluster] = []
        self.debug_mode = True

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
        for resolution_parameter in [1, 2, 3]:
            self.generate_graded_clusters(resolution_parameter)
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
        # Initiate the middle of the cluster as the initial cluster's centroid
        initial_centroid = cluster.default_centroid
        # Rank words by their distance to the centroid:
        cluster.df["centroid_distance"] = cluster.df["vector"].apply(lambda v: cosine_distance(v, initial_centroid))
        cluster.df.sort_values("centroid_distance", inplace=True)
        cluster = self.optimize_centroid_phys(cluster)
        # Generate list of optional sub_clusters within the cluster:
        # sub_clusters = []
        # for sub_cluster_size in range(1,len(cluster.df)+1):
        #     sub_df = cluster.df.iloc[0:sub_cluster_size, :]
        #     sub_cluster = Cluster(id=sub_cluster_size, df=sub_df)
        #     # For each sub_cluster, optimize it's centroid and grade it:
        #     if method == 'Geometric':
        #         optimized_sub_cluster = self.optimize_centroid_lin(sub_cluster)
        #     elif method == 'Physical':
        #         optimized_sub_cluster = self.optimize_centroid_phys(sub_cluster)
        #     # noinspection PyUnboundLocalVariable
        #     sub_clusters.append(optimized_sub_cluster)
        # best_cluster = max(sub_clusters)
        return cluster

    def color2force(self, centroid, row):
        if row["color"] == self.team_color.opponent.as_card_color:
            return opponent_force(centroid, row["vector"])
        elif row["color"] == CardColor.BLACK:
            return black_force(centroid, row["vector"])
        elif row["color"] == CardColor.GRAY:
            return gray_force(centroid, row["vector"])
        elif row["color"] == self.team_color.as_card_color:
            return friendly_force(centroid, row["vector"])
        else:
            raise ValueError(f"color{row['color']} is not a valid color")

    def board_df2nodes(self, centroid: np.array):  # -> List[Tuple[np.array, float], ...]:
        relevant_df = self.board_data[self.board_data["is_revealed"] == False]
        relevant_df["force"] = relevant_df.apply(lambda row: self.color2force(centroid, row), axis=1)
        relevant_df = relevant_df[["vector", "force"]]
        return list(relevant_df.itertuples(index=False, name=None))

    def optimize_centroid_lin(self, cluster: Cluster) -> Cluster:
        return cluster

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

    # MIN_BLACK_DISTANCE = 0.5
    # MIN_SELF_BLACK_DELTA = 0.3
    # MIN_SELF_OPPONENT_DELTA = 0.2
    # MIN_SELF_GRAY_DELTA = 0.1
    # MAX_SELF_DISTANCE = 0.6
    # OPPONENT_FORCE_CUTOFF = 0.4
    # OPPONENT_FORCE_FACTOR = 10
    # FRIENDLY_FORCE_CUTOFF = 0.4
    # FRIENDLY_FORCE_FACTOR = 1
    # FRIENDLY_FORCE_FACTOR = 0.5
    # BLACK_FORCE_FACTOR = 2

    def update_distances(self, centroid):
        # relevant_idx = self.board_data['is_revealed'].apply(lambda x: not x)
        # relevant_rows = self.board_data[relevant_idx]
        # self.board_data.loc[relevant_rows, 'distance_to_centroid'] =\
        # self.board_data.loc[relevant_rows, 'vector'].apply(lambda v: cosine_distance(v, centroid))
        self.board_data["distance_to_centroid"] = self.board_data["vector"].apply(
            lambda v: cosine_distance(v, centroid)
        )

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
            and (distance2black - max_distance2own > MIN_SELF_OPPONENT_DELTA)
            and (min(distances2gray) - max_distance2own > MIN_SELF_GRAY_DELTA)
            and (max_distance2own < MAX_SELF_DISTANCE)
        ):
            return True
        else:
            return False

    def clean_cluster(self, cluster: Cluster):
        cluster.df["centroid_distance"] = cluster.df["vector"].apply(lambda v: cosine_distance(v, cluster.centroid))
        max_distance = max(cluster.df["centroid_distance"])
        rouge_word = (cluster.df["centroid_distance"] > MAX_SELF_DISTANCE) | (
            cluster.df["centroid_distance"] != max_distance
        )
        cluster.df = cluster.df[rouge_word]
        cluster.centroid = cluster.default_centroid

    def optimize_centroid_phys(self, cluster: Cluster) -> Cluster:
        # temp_board = self.board_data.copy()
        # temp_board['centroid_distance'] = cluster.df['vector'].apply(lambda v: cosine_distance(v, centroid))
        # temp_board['in_current_cluster'] = temp_board.index.map(lambda x: x in cluster.df.index)
        # for i in range(100):
        #     self.clean_cluster(cluster)
        #     if self.debug_mode is True:
        #         self.draw_guesser_view(cluster.centroid)
        #     if self.optimization_break_condition(cluster):
        #         break
        #     self.clean_cluster(cluster)
        #     nodes = self.board_df2nodes(cluster.centroid)
        #     cluster.centroid = step_from_forces(cluster.centroid, nodes, arc_radians=0.001)
        cluster.centroid = cluster.default_centroid
        return cluster

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
        # closest_opponent_card = self.model.most_similar_to_given("king", ["queen", "prince"])

    def draw_guesser_view(self, centroid: np.array):
        self.update_distances(centroid)
        temp_df = self.unrevealed_cards.sort_values("distance_to_centroid")

        plt.bar(
            x=temp_df.index,
            height=temp_df["distance_to_centroid"],
            color=temp_df["color"].apply(lambda x: x.value.lower()),
        )
        ax = plt.gca()
        plt.setp(ax.get_xticklabels(), rotation="vertical")
        plt.show()
        # n = len(relevant_df)
        # G = nx.Graph()
        # G.add_node('centroid', color='green')
        # nodes_list = [(index, {'color': row['color'].value.lower()}) for index, row in relevant_df.iterrows()]
        # G.add_nodes_from(nodes_list)
        # edges_list = [('centroid', index, (1+row['distance_to_centroid'])**5) for index, row in relevant_df.iterrows()]
        # G.add_weighted_edges_from(edges_list)
        # # relevant_df['source'] = 'centroid_node'
        # # relevant_df['target'] = relevant_df.index
        # # relevant_df['color'] = relevant_df['color'].apply(lambda x: x.value.lower())
        # # relevant_df.rename(columns={'distance_to_centroid': 'length'}, inplace=True)
        # # relevant_df['length'] = relevant_df['length'].apply(lambda x: 1 / x**3)
        # # relevant_df = relevant_df[['source', 'target', 'length', 'color']]
        # pos = nx.spring_layout(G)
        # nx.draw(G, pos, with_labels=True)
        # plt.show()
        # render(G)
        # print('plottet')

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
                louvain_weight = distance ** (10 * resolution_parameter)
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

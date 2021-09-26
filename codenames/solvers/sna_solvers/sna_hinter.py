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

from codenames.game.base import TeamColor, Hint, Board, HinterGameState, CardColor, WordGroup
from codenames.game.player import Hinter
from codenames.solvers.naive.naive_hinter import Proposal, calculate_proposal_grade
from codenames.solvers.utils.algebra import cosine_distance, single_gram_schmidt
from language_data.model_loader import load_language


@dataclass
class ForceNode:
    force_origin: np.array
    force_sign: True
    force_size: Optional[float] = None


log = logging.getLogger(__name__)
MIN_SELF_BLACK_DELTA = 0.07
MIN_SELF_OPPONENT_DELTA = 0.04
MIN_SELF_GRAY_DELTA = 0.01
MAX_SELF_DISTANCE = 0.225
OPPONENT_FORCE_CUTOFF = 0.275
OPPONENT_FORCE_FACTOR = 1.6
FRIENDLY_FORCE_CUTOFF = 0.2
FRIENDLY_FORCE_FACTOR = 1
BLACK_FORCE_FACTOR = 2
GRAY_FORCE_FACTOR = 1.2
EPSILON = 0.001

BANNED_WORDS = {"slackerjack"}
Similarity = Tuple[str, float]
# Test


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


def step_away(starting_point: np.array, step_away_from: np.array, arc_radians: float) -> np.array:
    step_away_from, normed_o = single_gram_schmidt(step_away_from, starting_point)

    original_phase = np.arccos(np.clip(starting_point.T @ step_away_from, -1.0, 1.0))

    rotated = step_away_from * np.cos(original_phase + arc_radians) + normed_o * np.sin(original_phase + arc_radians)

    rotated_original_size = rotated * np.linalg.norm(starting_point)

    return rotated_original_size


def step_towards(starting_point: np.array, step_away_from: np.array, arc_radians: float) -> np.array:
    return step_away(starting_point, step_away_from, -arc_radians)


def sum_forces(starting_point: np.array, nodes) -> np.array:  # : List[ForceNode,...]
    total_force = np.zeros(nodes[0].force_origin.shape)
    for node in nodes:
        rotated = step_away(starting_point, node.force_origin, node.force_size * EPSILON)
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
    def words(self) -> WordGroup:
        return tuple(self.df.index)

    @property
    def default_centroid(self) -> np.array:
        non_normalized_v = np.mean(self.df["vector_normed"])
        normalized_v = non_normalized_v / np.linalg.norm(non_normalized_v)
        return normalized_v

    def update_distances(self):
        self.df["centroid_distance"] = self.df["vector"].apply(lambda v: cosine_distance(v, self.centroid))

    def sort_by_distances(self):
        self.df.sort_values("centroid_distance", inplace=True)

    def reset(self):
        self.centroid = self.default_centroid
        self.update_distances()
        self.sort_by_distances()

    def __gt__(self, cluster_2):
        return self.grade > cluster_2.grade

    def __lt__(self, cluster_2):
        return self.grade < cluster_2.grade


class SnaHinter(Hinter):
    def __init__(self, name: str, team_color: TeamColor, debug_mode=True, physics_optimization=True):
        super().__init__(name=name, team_color=team_color)
        self.model: Optional[KeyedVectors] = None
        self.language_length: Optional[int] = None
        self.board_data: Optional[pd.DataFrame] = None
        self.graded_proposals: List[Cluster] = []
        self.debug_mode = debug_mode
        self.physics_optimization = physics_optimization
        self.game_state = None

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
    def own_unrevealed_cards(self) -> pd.DataFrame:
        own_unrevealed_idx = (self.board_data.is_revealed == False) & (  # noqa: E712
            self.board_data.color == self.team_color.as_card_color
        )
        return self.board_data[own_unrevealed_idx]

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

    def update_reveals(self, game_state):
        mapper = {card.word: card.revealed for card in game_state.board}
        self.board_data["is_revealed"] = self.board_data.index.map(mapper)

    def pick_hint(self, game_state: HinterGameState) -> Hint:
        self.game_state = game_state
        self.update_reveals(game_state)
        graded_proposals = self.generate_graded_proposals()
        best_proposal = graded_proposals[0]
        draw_cluster = Cluster(
            -1,
            self.board_data[self.board_data.index.isin(best_proposal.word_group)],
            self.model.get_vector(best_proposal.hint_word),
        )
        self.draw_guesser_view(draw_cluster, best_proposal.hint_word, self.model.get_vector(best_proposal.hint_word))
        hint = Hint(best_proposal.hint_word, best_proposal.card_count)
        return hint

    def generate_graded_proposals(self, resolution_parameter=1):
        self.divide_to_clusters(df=self.own_unrevealed_cards, resolution_parameter=resolution_parameter)
        graded_proposals = []
        unique_clusters_ids = pd.unique(self.own_unrevealed_cards.cluster)
        for cluster_id in unique_clusters_ids:
            df = self.own_unrevealed_cards.loc[self.own_unrevealed_cards.cluster == cluster_id, :]
            cluster = Cluster(id=cluster_id, df=df.copy(deep=True))
            proposal = self.cluster2proposal(cluster)
            graded_proposals.append(proposal)
        graded_proposals.sort(key=lambda c: -c.grade)
        return graded_proposals

    def cluster2proposal(self, cluster: Cluster):
        self.optimize_cluster(cluster)
        similarities: List[Similarity] = self.model.most_similar(cluster.centroid, topn=100)
        best_proposal = self.pick_best_similarity(
            similarities=similarities,
            words_to_filter_out={*self.board_data.index.to_list(), *self.game_state.given_hint_words},
        )
        return best_proposal

    def pick_best_similarity(
        self, similarities: List[Similarity], words_to_filter_out: Iterable[str]
    ) -> Optional[Proposal]:
        words_to_filter_out = {word.lower() for word in words_to_filter_out}
        filtered_proposals = []
        for similarity in similarities:
            word, grade = similarity
            word = word.lower()
            if should_filter_word(word, words_to_filter_out):
                continue
            vector = self.model.get_vector(word)
            proposal = self.vec2proposal(word, vector)
            filtered_proposals.append(proposal)
        if len(filtered_proposals) == 0:
            return None
        best_proposal = min(filtered_proposals, key=lambda p: -p.grade)
        return best_proposal

    def vec2proposal(self, word, vector) -> Proposal:
        self.update_distances(vector)
        temp_df = self.unrevealed_cards.sort_values("distance_to_centroid")
        centroid_to_black = np.min(  # This min is required for the float type
            cosine_distance(vector, temp_df[temp_df["color"] == CardColor.BLACK]["vector"])
        )
        centroid_to_gray = np.min(cosine_distance(vector, temp_df[temp_df["color"] == CardColor.GRAY]["vector"]))
        centroid_to_opponent = np.min(
            cosine_distance(
                vector,
                temp_df[temp_df["color"] == self.team_color.opponent.as_card_color]["vector"],
            )
        )

        bad_cards_limitation = np.min(
            [
                centroid_to_black - MIN_SELF_BLACK_DELTA,
                centroid_to_gray - MIN_SELF_GRAY_DELTA,
                centroid_to_opponent - MIN_SELF_OPPONENT_DELTA,
            ]
        )

        chosen_cards = temp_df[
            (temp_df["distance_to_centroid"] < bad_cards_limitation)
            & (temp_df["distance_to_centroid"] < MAX_SELF_DISTANCE)
            & (temp_df["color"] == self.team_color.as_card_color)
        ]

        distance_group = np.max(chosen_cards["distance_to_centroid"])

        if self.debug_mode is True:
            draw_cluster = Cluster(0, chosen_cards)
            draw_cluster.reset()
            self.draw_guesser_view(cluster=draw_cluster, word=word, vector=vector)

        proposal = Proposal(
            word_group=chosen_cards.index.to_list(),
            hint_word=word,
            hint_word_frequency=0,
            distance_group=distance_group,
            distance_gray=centroid_to_gray,
            distance_opponent=centroid_to_opponent,
            distance_black=centroid_to_black,
        )
        proposal.grade = calculate_proposal_grade(proposal)

        return proposal

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
        tuples_list = list(relevant_df.itertuples(index=False, name=None))
        nodes_list = [ForceNode(force_origin=element[0], force_size=element[1]) for element in tuples_list]
        return nodes_list

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

    def optimize_cluster(self, cluster: Cluster) -> Cluster:
        cluster.reset()
        if self.debug_mode is True:
            self.draw_guesser_view(cluster)
        for i in range(100):
            self.clean_cluster(cluster)
            if self.debug_mode is True:
                self.draw_guesser_view(cluster)
            if self.optimization_break_condition(cluster):
                break
            if self.physics_optimization:
                nodes = self.board_df2nodes(cluster.centroid)
                cluster.centroid = step_from_forces(cluster.centroid, nodes, arc_radians=5e-2)
            if self.debug_mode is True:
                self.draw_guesser_view(cluster)
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

    def update_distances(self, centroid):
        self.board_data["distance_to_centroid"] = self.board_data["vector"].apply(
            lambda v: cosine_distance(v, centroid)
        )

    def clean_cluster(self, cluster: Cluster):
        cluster.update_distances()
        max_distance = max(cluster.df["centroid_distance"])
        central_words = (cluster.df["centroid_distance"] < MAX_SELF_DISTANCE) | (
            cluster.df["centroid_distance"] != max_distance
        )
        cluster.df = cluster.df[central_words]
        cluster.centroid = cluster.default_centroid

    def draw_centroid_distances(self, ax, cluster: Cluster, centroid=None, title=None):
        if centroid is None:
            self.update_distances(cluster.centroid)
        else:
            self.update_distances(centroid)

        temp_df = self.unrevealed_cards.sort_values("distance_to_centroid")
        colors = temp_df["color"].apply(lambda x: x.value.lower())
        temp_df["is_in_cluster"] = temp_df.index.isin(cluster.df.index)
        edge_color = temp_df["is_in_cluster"].apply(lambda x: "yellow" if x else "black")
        line_width = temp_df["is_in_cluster"].apply(lambda x: 3 if x else 1)
        ax.bar(
            x=temp_df.index,
            height=temp_df["distance_to_centroid"],
            color=colors,
            edgecolor=edge_color,
            linewidth=line_width,
        )
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.set_title(title)

    def draw_guesser_view(self, cluster: Cluster, word=None, vector=None):
        if word is None:
            fig, ax = plt.subplots(1, 1, figsize=(15, 8))
            self.draw_centroid_distances(ax, cluster, title="Cluster centroid")
            plt.show()
        else:
            fig, ax = plt.subplots(2, 1, figsize=(15, 8))
            self.draw_centroid_distances(ax[0], cluster, centroid=vector, title=word)
            self.draw_centroid_distances(ax[1], cluster, title="Cluster centroid")
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

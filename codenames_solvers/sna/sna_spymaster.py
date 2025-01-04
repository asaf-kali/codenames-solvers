# type: ignore
import itertools
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import community
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from codenames.classic.color import ClassicColor
from codenames.generic.board import Board, WordGroup
from codenames.generic.move import Clue
from codenames.generic.player import Spymaster
from codenames.generic.state import SpymasterState
from gensim.models import KeyedVectors
from pandas import Series

from codenames_solvers.models import load_language
from codenames_solvers.naive.naive_spymaster import (
    Proposal,
    default_proposal_grade_calculator,
)
from codenames_solvers.utils import RUN_ID, get_exports_folder
from codenames_solvers.utils.algebra import cosine_distance, single_gram_schmidt

plt.style.use("fivethirtyeight")


@dataclass
class ForceNode:
    force_origin: np.array
    force_sign: True
    force_size: Optional[float] = None


log = logging.getLogger(__name__)
MIN_SELF_ASSASSIN_DELTA = 0.07
MIN_SELF_OPPONENT_DELTA = 0.04
MIN_SELF_NEUTRAL_DELTA = 0.01
MAX_SELF_DISTANCE = 0.235
OPPONENT_FORCE_CUTOFF = 0.275
OPPONENT_FORCE_FACTOR = 1.6
FRIENDLY_FORCE_CUTOFF = 0.2
FRIENDLY_FORCE_FACTOR = 1
ASSASSIN_FORCE_FACTOR = 2
NEUTRAL_FORCE_FACTOR = 1.2
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
    cos_phase = (starting_point.T @ step_away_from) / (np.linalg.norm(step_away_from) * np.linalg.norm(starting_point))

    original_phase = np.arccos(np.clip(cos_phase, -1.0, 1.0))

    step_away_from, normed_o = single_gram_schmidt(step_away_from, starting_point)

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
        np.set_printoptions(precision=6)
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
        # Parabola with 0 at d=0, 1 at d=FRIENDLY_FORCE_CUTOFF and else otherwise:
        return FRIENDLY_FORCE_FACTOR * (
            1 - (d / FRIENDLY_FORCE_CUTOFF - 1) ** 2
        )  # FRIENDLY_FORCE_FACTOR * d / FRIENDLY_FORCE_CUTOFF


def repelling_force(d, cutoff_distance, factor):
    if d > OPPONENT_FORCE_CUTOFF:
        return 0
    else:
        a = factor / (factor - 1) * cutoff_distance
        return a / (d + a / factor)


def opponent_force(d: float):
    return repelling_force(d, OPPONENT_FORCE_CUTOFF, OPPONENT_FORCE_FACTOR)


def neutral_force(d):
    return repelling_force(d, OPPONENT_FORCE_CUTOFF, NEUTRAL_FORCE_FACTOR)


def black_force(d):
    return repelling_force(d, OPPONENT_FORCE_CUTOFF, ASSASSIN_FORCE_FACTOR)


def _format_word(word: str) -> str:
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
        mean = np.mean(self.df["vector_normed"])
        normalized_mean = mean / np.linalg.norm(mean)
        return normalized_mean

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


class SNASpymaster(Spymaster):
    def __init__(self, name: str, debug_mode=False, physics_optimization=True):
        super().__init__(name=name)
        self.model: Optional[KeyedVectors] = None
        self.language_length: Optional[int] = None
        self.board_data: Optional[pd.DataFrame] = None
        self.graded_proposals: List[Cluster] = []
        self.debug_mode = debug_mode
        self.physics_optimization = physics_optimization
        self.game_state: Optional[SpymasterState] = None

    def on_game_start(self, board: Board):
        self.model = load_language(language=board.language)  # type: ignore
        self.language_length = len(self.model.index_to_key)
        all_words = [_format_word(word) for word in board.all_words]
        vectors = self.model[all_words]
        vectors_list = list(vectors)
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
        return self.board_data[self.board_data["is_revealed"] == False]  # noqa: E712

    @property
    def own_unrevealed_cards(self) -> pd.DataFrame:
        own_unrevealed_idx = (self.board_data.is_revealed == False) & (  # noqa: E712
            self.board_data.color == self.team.as_card_color
        )
        return self.board_data[own_unrevealed_idx]

    @property
    def opponent_cards(self) -> pd.DataFrame:
        return self.board_data[self.board_data["color"] == self.team.opponent.as_card_color]

    @property
    def neutral_cards(self) -> pd.DataFrame:
        return self.board_data[self.board_data["color"] == ClassicColor.NEUTRAL]

    @property
    def own_cards(self) -> pd.DataFrame:
        return self.board_data[self.board_data["color"] == self.team]

    @property
    def black_card(self) -> pd.DataFrame:
        return self.board_data[self.board_data["color"] == ClassicColor.ASSASSIN]

    @property
    def bad_cards(self) -> pd.DataFrame:
        return self.board_data[
            self.board_data["color"].isin(
                [ClassicColor.NEUTRAL, ClassicColor.ASSASSIN, self.team.opponent.as_card_color]
            )
        ]

    def update_reveals(self, game_state):
        mapper = {card.word: card.revealed for card in game_state.board}
        self.board_data["is_revealed"] = self.board_data.index.map(mapper)

    def give_clue(self, game_state: SpymasterState) -> Clue:
        self.game_state = game_state
        self.update_reveals(game_state)
        graded_proposals = self.generate_graded_proposals()
        graded_proposals.sort(key=lambda p: -p.grade)
        best_n_repr = "\n".join(str(p) for p in graded_proposals[:3])
        log.info(f"Best proposals: \n{best_n_repr}")
        best_proposal = graded_proposals[0]
        # draw_cluster = Cluster(
        #     -1,
        #     self.board_data[self.board_data.index.isin(best_proposal.word_group)],
        #     self.model.get_vector(best_proposal.clue_word),
        # )
        # self.draw_operative_view(draw_cluster, best_proposal.clue_word, self.model.get_vector(best_proposal.clue_word))
        clue = Clue(word=best_proposal.clue_word, card_amount=best_proposal.card_count)
        return clue

    def generate_graded_proposals(self, resolution_parameter=1):
        self.divide_to_clusters(df=self.own_unrevealed_cards, resolution_parameter=resolution_parameter)
        graded_proposals = []
        unique_clusters_ids = pd.unique(self.own_unrevealed_cards.cluster)
        for cluster_id in unique_clusters_ids:
            df = self.own_unrevealed_cards.loc[self.own_unrevealed_cards.cluster == cluster_id, :]
            cluster = Cluster(id=cluster_id, df=df.copy(deep=True))
            proposal = self.proposal_from_cluster(cluster)
            graded_proposals.append(proposal)
            draw_cluster = Cluster(
                -1,
                self.board_data[self.board_data.index.isin(proposal.word_group)],
                self.model.get_vector(proposal.clue_word),
            )
            self.draw_operative_view(draw_cluster, proposal.clue_word, self.model.get_vector(proposal.clue_word))
        graded_proposals.sort(key=lambda c: -c.grade)
        return graded_proposals

    def proposal_from_cluster(self, cluster: Cluster):
        self.optimize_cluster(cluster)
        similarities: List[Similarity] = self.model.most_similar(cluster.centroid, topn=100)
        best_proposal = self.pick_best_similarity(
            similarities=similarities,
            words_to_filter_out=self.game_state.illegal_words,
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
            proposal = self.proposal_from_word_vector(word, vector)
            filtered_proposals.append(proposal)
        if len(filtered_proposals) == 0:
            return None
        best_proposal = min(filtered_proposals, key=lambda p: -p.grade)
        return best_proposal

    def proposal_from_word_vector(self, word: str, vector: np.ndarray) -> Proposal:
        self.update_distances(vector)
        temp_df = self.unrevealed_cards.sort_values("distance_to_centroid")
        centroid_to_black = np.min(  # This min is required for the float type
            cosine_distance(vector, temp_df[temp_df["color"] == ClassicColor.ASSASSIN]["vector"])
        )
        centroid_to_neutral = np.min(
            cosine_distance(vector, temp_df[temp_df["color"] == ClassicColor.NEUTRAL]["vector"])
        )
        centroid_to_opponent = np.min(
            cosine_distance(
                vector,
                temp_df[temp_df["color"] == self.team.opponent.as_card_color]["vector"],
            )
        )

        bad_cards_limitation = np.min(
            [
                centroid_to_black - MIN_SELF_ASSASSIN_DELTA,
                centroid_to_neutral - MIN_SELF_NEUTRAL_DELTA,
                centroid_to_opponent - MIN_SELF_OPPONENT_DELTA,
            ]
        )

        chosen_cards = temp_df[
            (temp_df["distance_to_centroid"] < bad_cards_limitation)
            & (temp_df["distance_to_centroid"] < MAX_SELF_DISTANCE)
            & (temp_df["color"] == self.team.as_card_color)
        ]

        distance_group = np.max(chosen_cards["distance_to_centroid"])

        if self.debug_mode is True:
            draw_cluster = Cluster(0, chosen_cards)
            draw_cluster.reset()
            self.draw_operative_view(cluster=draw_cluster, word=word, vector=vector)

        proposal = Proposal(
            word_group=chosen_cards.index.to_list(),
            clue_word=word,
            clue_word_frequency=0,
            distance_group=distance_group,
            distance_neutral=centroid_to_neutral,
            distance_opponent=centroid_to_opponent,
            distance_black=centroid_to_black,
        )
        proposal.grade = default_proposal_grade_calculator(proposal)

        return proposal

    def force_from_color(self, centroid: np.ndarray, card_row: Series):
        vector, card_color = card_row["vector"], card_row["color"]
        d = cosine_distance(centroid, vector)
        if card_color == self.team.opponent.as_card_color:
            return opponent_force(d)
        elif card_color == ClassicColor.ASSASSIN:
            return black_force(d)
        elif card_color == ClassicColor.NEUTRAL:
            return neutral_force(d)
        elif card_color == self.team.as_card_color:
            return friendly_force(d)
        else:
            raise ValueError(f"color{card_row['color']} is not a valid color")

    def board_df2nodes(self, centroid: np.array):  # -> List[Tuple[np.array, float], ...]:
        relevant_df = self.board_data[self.board_data["is_revealed"] == False]  # noqa: E712
        relevant_df["force"] = relevant_df.apply(lambda row: self.force_from_color(centroid, row), axis=1)
        relevant_df = relevant_df[["vector", "force"]]
        tuples_list = list(relevant_df.itertuples(index=False, name=None))
        nodes_list = [
            ForceNode(force_origin=element[0], force_sign=True, force_size=element[1]) for element in tuples_list
        ]
        return nodes_list

    def optimization_break_condition(self, cluster: Cluster) -> bool:
        self.update_distances(cluster.centroid)
        distances2opponent = self.extract_centroid_distances(self.team.opponent.as_card_color)
        distances2own = self.extract_centroid_distances(self.team.as_card_color)
        distances2own = distances2own[distances2own.index.isin(cluster.df.index.to_list())]
        distance2black = self.extract_centroid_distances(ClassicColor.ASSASSIN)
        distances2neutral = self.extract_centroid_distances(ClassicColor.NEUTRAL)
        max_distance2own = max(distances2own)
        if (
            (min(distances2opponent) - max_distance2own > MIN_SELF_OPPONENT_DELTA)
            and (distance2black[0] - max_distance2own > MIN_SELF_OPPONENT_DELTA)
            and (min(distances2neutral) - max_distance2own > MIN_SELF_NEUTRAL_DELTA)
            and (max_distance2own < MAX_SELF_DISTANCE)
        ):
            return True
        return False

    def optimize_cluster(self, cluster: Cluster) -> Cluster:
        cluster.reset()
        if self.debug_mode is True:
            self.draw_operative_view(cluster)
        for _ in range(100):
            self.clean_cluster(cluster)
            if self.debug_mode is True:
                self.draw_operative_view(cluster)
            if self.optimization_break_condition(cluster):
                break
            if self.physics_optimization:
                nodes = self.board_df2nodes(cluster.centroid)
                cluster.centroid = step_from_forces(cluster.centroid, nodes, arc_radians=5e-2)
            if self.debug_mode is True:
                self.draw_operative_view(cluster)
        return cluster

    def extract_centroid_distances(self, color: ClassicColor):
        relevant_df = self.board_data[self.board_data["is_revealed"] == False]  # noqa: E712
        if color == ClassicColor.NEUTRAL:
            color_rows = relevant_df.color == ClassicColor.NEUTRAL
        elif color == ClassicColor.ASSASSIN:
            color_rows = relevant_df.color == ClassicColor.ASSASSIN
        elif color == self.team.opponent.as_card_color:
            color_rows = relevant_df.color == self.team.opponent.as_card_color
        elif color == self.team.as_card_color:
            color_rows = relevant_df.color == self.team.as_card_color
        elif color == ClassicColor.BAD:  # TODO: What is this?
            color_rows = relevant_df.color.isin(
                [ClassicColor.NEUTRAL, ClassicColor.ASSASSIN, self.team.opponent.as_card_color]
            )
        else:
            raise ValueError(f"No such color as {color}")
        return relevant_df.loc[color_rows, "distance_to_centroid"]

    def update_distances(self, centroid):
        self.board_data["distance_to_centroid"] = self.board_data["vector"].apply(
            lambda v: cosine_distance(v, centroid)
        )

    @staticmethod
    def clean_cluster(cluster: Cluster):
        cluster.update_distances()
        centroid_distances = cluster.df["centroid_distance"]
        max_distance = max(centroid_distances)
        central_words = (centroid_distances < MAX_SELF_DISTANCE) | (centroid_distances != max_distance)
        cluster.df = cluster.df[central_words]
        cluster.centroid = cluster.default_centroid

    def draw_centroid_distances(self, ax, cluster: Cluster, centroid=None, title=None):
        if centroid is None:
            self.update_distances(cluster.centroid)
        else:
            self.update_distances(centroid)

        temp_df = self.unrevealed_cards.sort_values("distance_to_centroid")
        temp_df["colors"] = temp_df["color"].apply(lambda x: x.value.lower())
        temp_df["is_in_cluster"] = temp_df.index.isin(cluster.df.index)
        temp_df["edge_color"] = temp_df["is_in_cluster"].apply(lambda x: "yellow" if x else "black")
        temp_df["line_width"] = temp_df["is_in_cluster"].apply(lambda x: 3 if x else 1)

        ax.bar(
            x=temp_df.index,
            height=temp_df["distance_to_centroid"],
            color=temp_df["colors"],
            edgecolor=temp_df["edge_color"],
            linewidth=temp_df["line_width"],
        )
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.set_title(title)
        file_name = f"{datetime.now().timestamp()}-{title}"
        if file_name != "no":
            export_folder = get_exports_folder("sna", RUN_ID)
            export_file = os.path.join(export_folder, file_name)
            temp_df.to_csv(f"{export_file}.csv")
            plt.savefig(f"{export_file}.png")
        plt.show()

    def draw_operative_view(self, cluster: Cluster, word=None, vector=None):
        if word is None:
            fig, ax = plt.subplots(1, 1, figsize=(15, 8))
            self.draw_centroid_distances(ax, cluster, title="Cluster centroid")
        else:
            fig, ax = plt.subplots(1, 1, figsize=(15, 8))
            self.draw_centroid_distances(ax, cluster, centroid=vector, title=f"clue word: {word}")
            # self.draw_centroid_distances(ax[1], cluster, title="Cluster centroid")

    def divide_to_clusters(self, df: pd.DataFrame, resolution_parameter=1):
        vis_graph = nx.Graph()
        words = df.index.to_list()
        vis_graph.add_nodes_from(words)
        louvain = nx.Graph(vis_graph)
        for word_couple in itertools.combinations(words, 2):
            v, u = _format_word(word_couple[0]), _format_word(word_couple[1])
            distance = self.model.similarity(v, u) + 1
            if distance > 1.1:
                vis_graph.add_edge(v, u, weight=distance)
            louvain_weight = distance ** (2 * resolution_parameter)
            louvain.add_edge(v, u, weight=louvain_weight)

        word_to_group: Dict[str, int] = community.best_partition(louvain)
        self.board_data.cluster = self.board_data.index.map(word_to_group)

# type: ignore
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Sequence, Union

import community
import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from codenames.game.base import TeamColor, Hint, Board, HinterGameState
from codenames.game.player import Hinter
from codenames.model_loader import load_language

log = logging.getLogger(__name__)
SIMILARITY_LOWER_BOUNDARY = 0.5

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


def single_gram_schmidt(v: np.array, u: np.array) -> Tuple[np.array, np.array]:
    v = v / np.linalg.norm(v)
    u = u / np.linalg.norm(u)

    projection_norm = u.T @ v

    o = u - projection_norm * v

    normed_o = o / np.linalg.norm(o)
    return v, normed_o


def step_away(step_away_from: np.array, starting_point: np.array, arc_radians: float) -> np.array:
    step_away_from, normed_o = single_gram_schmidt(step_away_from, starting_point)

    original_phase = starting_point.T @ step_away_from

    rotated = step_away_from * np.cos(original_phase + arc_radians) + normed_o * np.sin(original_phase + arc_radians)

    rotated_original_size = rotated * np.linalg.norm(starting_point)

    return rotated_original_size


def cosine_similarity_with_vectors(u: np.array, v: Sequence[np.array]) -> np.array:
    u = u / np.linalg.norm(u)
    v_list = [vec / np.linalg.norm(vec) for vec in v]
    return np.array([u.T @ vec for vec in v_list])


def cosine_similarity_with_vector(u: np.array, v: np.array) -> float:
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    return u.T @ v


def cosine_similarity(u: np.array, v: Union[np.array, Sequence[np.array]]) -> Union[float, np.array]:
    if isinstance(v, pd.Series):
        return cosine_similarity_with_vectors(u, v)
    else:
        return cosine_similarity_with_vector(u, v)


def cosine_distance(u: np.array, v: np.array) -> np.array:
    return 1 - cosine_similarity(u, v)


def format_word(word: str) -> str:
    return word.replace(" ", "_").replace("-", "_").strip()


@dataclass
class Cluster:
    id: int
    # centroid: np.array
    rows: pd.DataFrame
    grade: float = 0

    @property
    def words(self) -> Tuple[str, ...]:
        return tuple(self.rows.index)

    @property
    def centroid(self) -> np.array:
        non_normalized_v = np.mean(self.rows["vector_normed"])
        normalized_v = non_normalized_v / np.linalg.norm(non_normalized_v)
        return normalized_v


class SnaHinter(Hinter):
    def __init__(self, name: str, team_color: TeamColor):
        super().__init__(name=name, team_color=team_color)
        self.model: Optional[KeyedVectors] = None
        self.language_length: Optional[int] = None
        self.board_data: Optional[pd.DataFrame] = None
        self.graded_clusters: List[Cluster] = []

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
            },
            index=board.all_words,
        )

    @property
    def opponent_cards(self) -> pd.DataFrame:
        return self.board_data[self.board_data.color == self.team_color.opponent]

    @property
    def gray_cards(self) -> pd.DataFrame:
        return self.board_data[self.board_data.color == "Gray"]

    @property
    def own_cards(self) -> pd.DataFrame:
        return self.board_data[self.board_data.color == self.team_color]

    @property
    def black(self) -> pd.DataFrame:
        return self.board_data[self.board_data.color == "Black"]

    def pick_hint(self, state: HinterGameState) -> Hint:
        self.board_data.is_revealed = state.board.all_reveals
        self.generate_graded_clusters()
        for cluster in self.graded_clusters:
            cluster_size = len(cluster.rows)
            centroid = self.optimize_centroid(cluster.centroid)
            similarities: List[Similarity] = self.model.most_similar(centroid)
            cluster_words = cluster.rows.index.to_list()
            best_similarity = pick_best_similarity(
                similarities=similarities, words_to_filter_out={*cluster_words, *state.given_hint_words}
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

    def generate_graded_clusters(self):
        unrevealed_index = (self.board_data.is_revealed == False) & (  # noqa: E712
            self.board_data.color == self.team_color.as_card_color
        )
        unrevealed_cards = self.board_data[unrevealed_index]
        self.divide_to_clusters(rows=unrevealed_cards)
        unrevealed_cards = self.board_data[unrevealed_index]  # Now updated with cluster columns
        self.graded_clusters = []
        unique_clusters_ids = pd.unique(unrevealed_cards.cluster)
        for cluster_id in unique_clusters_ids:
            rows = unrevealed_cards[unrevealed_cards.cluster == cluster_id]
            cluster = Cluster(id=cluster_id, rows=rows)
            self.grade_cluster(cluster)
            self.graded_clusters.append(cluster)
        self.graded_clusters.sort(key=lambda c: -c.grade)

    def optimize_centroid(self, centroid: np.array) -> np.array:
        return centroid

    # flake8: noqa: F841
    def grade_cluster(self, cluster: Cluster) -> float:
        distances = cosine_distance(cluster.centroid, cluster.rows.vector)
        # centroid_to_black = cosine_distance(
        #     cluster.centroid, self.board_data[self.board_data.color == "Black"]["vector"]
        # )
        # centroid_to_gray = np.min(
        #     cosine_distance(cluster.centroid, self.board_data[self.board_data.color == "Gray"]["vector"])
        # )
        # centroid_to_opponent = 0
        return np.mean(distances)  # type: ignore
        # closest_opponent_card = self.model.most_similar_to_given("king", ["queen", "prince"])

    def divide_to_clusters(self, rows: pd.DataFrame):
        board_size = len(rows)
        vis_graph = nx.Graph()
        words = rows.index.to_list()
        vis_graph.add_nodes_from(words)
        louvain = nx.Graph(vis_graph)
        for i in range(board_size):
            v = format_word(words[i])
            for j in range(i + 1, board_size):
                u = format_word(words[j])
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

from manim import *
import numpy as np
import random
from typing import Callable, List
from codenames.solvers.utils.algebra import single_gram_schmidt, geodesic
from codenames.solvers.sna_solvers.sna_hinter import step_from_forces, black_force, gray_force, opponent_force, friendly_force

def random_ints(n: int, k: int) -> List[int]:
    ints_list = [random.randint(0, n) for i in range(k)]
    return ints_list


def generate_random_subsets(elements_list: List, average_subset_size: int) -> List[List]:
    k = len(elements_list)
    n = int(k / average_subset_size)
    subsets_map = random_ints(n, k)
    zip_iterator = zip(elements_list, subsets_map)
    mapper = list(zip_iterator)
    subsets_list = []
    for s in set(subsets_map):
        subset_elements = [element[0] for element in mapper if element[1] == s]
        subsets_list.append(subset_elements)
    return subsets_list


def generate_cluster_connections(cluster_vectors: List[np.ndarray]) -> List[ParametricFunction]:
    n = len(cluster_vectors)
    connections_list = []
    if n < 2:
        return connections_list
    for i in range(n - 1):
        for j in range(i + 1, n):
            geodesic_function = geodesic(cluster_vectors[i], cluster_vectors[j])
            connection = ParametricFunction(geodesic_function, t_range=[0, 1])
            connections_list.append(connection)
    return connections_list


def generate_random_connections(vectors_list, average_cluster_size=2):
    vectors_clusters = generate_random_subsets(vectors_list, average_subset_size=average_cluster_size)
    connections_list = []
    for cluster in vectors_clusters:
        cluster_connections = generate_cluster_connections(cluster)
        connections_list.extend(cluster_connections)
    return connections_list


def polar_to_cartesian(r, phi, theta):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.array([x, y, z])


def text_len_to_time(text, min_time=2, seconds_per_char=0.1):
    n = len(text)
    return np.max([min_time, n * seconds_per_char])

def generate_nodes(centroid, attractors, repellers):
    return None

def record_trajectory(starting_point, attractors, repellers):
    trajectory = [starting_point]
    centroid = starting_point
    for i in range(100):
        nodes = generate_nodes(centroid, attractors, repellers)
        centroid = step_from_forces(centroid, nodes, arc_radians=0.01)
        trajectory.append(centroid)
    return trajectory

# def connecting_line(v, u):
#   c = v.T @ u
#   f = lambda t: (t*v+(1-t)*u) / np.sqrt(t**2+(1-t**2) + 2*t*(1-t)*c)
#   return f

SPHERE_RADIUS = 1
FONT_SIZE_LABELS = 12
FONT_SIZE_TEXT = 12
ARROWS_THICKNESS = 0.001
DOT_SIZE = 0.2
KING_VEC = polar_to_cartesian(SPHERE_RADIUS, 0.5 * PI, 0)
QUEEN_VEC = polar_to_cartesian(SPHERE_RADIUS, 0.5 * PI, 0.2 * PI)
BEACH_VEC = polar_to_cartesian(SPHERE_RADIUS, 0.70 * PI, 1.17 * PI)
PARK_VEC = polar_to_cartesian(SPHERE_RADIUS, 0.75 * PI, 1.25 * PI)
JUPITER_VEC = polar_to_cartesian(SPHERE_RADIUS, 0.2 * PI, -0.3 * PI)
NEWTON_VEC = polar_to_cartesian(SPHERE_RADIUS, 0.27 * PI, -0.11 * PI)
TEACHER_VEC = polar_to_cartesian(SPHERE_RADIUS, 0.3 * PI, -0.2 * PI)

vectors_list = [KING_VEC, QUEEN_VEC, BEACH_VEC, PARK_VEC, JUPITER_VEC, NEWTON_VEC, TEACHER_VEC]

generate_random_connections(vectors_list)


labels_list = ["king", "queen", "beach", "park", "jupiter", "newton", "teacher"]
list_len = len(vectors_list)
sna_connections_list = [
    ParametricFunction(geodesic(KING_VEC, QUEEN_VEC), t_range=[0, 1]),
    ParametricFunction(geodesic(BEACH_VEC, PARK_VEC), t_range=[0, 1]),
    ParametricFunction(geodesic(JUPITER_VEC, NEWTON_VEC), t_range=[0, 1]),
    ParametricFunction(geodesic(TEACHER_VEC, JUPITER_VEC), t_range=[0, 1]),
    ParametricFunction(geodesic(NEWTON_VEC, TEACHER_VEC), t_range=[0, 1]),
]

script = {
    "The algorithm uses...": "The Kalirmoz algorithm uses a Word2Vec model for the linguistic knowledge",
    "In a nutshell...": "In a nutshell, the Word2Vec assign each word with an n-dimensional\n"
    "vector (usually n=50, 100, 300), in a way such that words that\n"
    "tent to appear in the same context have small angle between them",
    "For the sake of...": "For the sake of this video, we will represent the words vectors\n"
    "as 3-dimensional vectors",
    "Here are some...": "Here are some words and their corresponding vectors.",
    "The word X...": "The word X is close to the words Y, Z and far from the word W\n"
    "as indeed semantically, the words X,Y,Z all appear in the\n"
    "of bla bla, while the word W usually appears in the context of\b"
    "blu blue",
    "In each turn...": "In each turn, the first task of the hinter is to find a\n"
    "proper subset of words (usually two to four words), on\n"
    "which to hint",
    "Two methods...": "Two methods of clustering where implemented.",
    "In the first cluste...": "In the first clustering method, the words are considered\n"
    "as nodes in a graph, with edges weights correlated to their\n"
    "cosine similarity",
    "This graph is divid...": "This graph is divided into communities using the louvain\n"
    "SNA algorithm, and each community is taken as an optional\n"
    "cluster of words to hint about.",
    "Here is an example...": "Here is an example of 25 words and their louvain\n" "clustering result:",
    "As can be seen...": "As can be seen, semantically close words are put within the\n" "same cluster.",
    "The second clusteri...": "The second clustering method is much simpler:",
    "Since there are...": "Since there are at most 9 cards to hint about, it is feasible\n"
    "to just iterate over all possible subsets and choose the best\n"
    "one.",
    "The second task...": "The second task of the hinter is to choose a hinting word for\n" "the cluster.",
    "In order to find...": "In order to find a hinting word for a cluster, the hinter\n"
    'generates a "centroid" vector for the cluster, to search real\n'
    "words near by.",
    "An initial centroid...": 'An initial "centroid" is proposed as the Center of Mass of\n' "the cluster's vectors",
    "Ideally, the centro...": "Ideally, the centroid would be close to all the cluster's words\n"
    'and far from words of other colors. (where "close" and "far")\n'
    "are considered in the cosine distance metric.",
    "to optimize the...": "To optimize the centroid, the words in the board (from\n"
    " all colors) are considered as a physical system, where\n"
    "every vector from the color of the hinter is an attractor,\n"
    "and every word from other color is a repeller.",
    "The centroid is the...": "The centroid is then being pushed and pulled by the words\n"
    "of the board until converging to a point where it is both\n"
    "far away from bad words, and close to close words.",
    "The attraction forc...": "The attraction force acts like a spring, where if the\n"
    'centroid is to far, the spring can be "torn" apart and is\n'
    "no longer considered as part of the cluster.",
    "This is done in ord...": "This is done in order to allow outliers in the cluster to be\n" "neglected.",
    "After convergence...": "After convergence, all there needs to be done is to pick up a\n"
    "word near-by the optimized cluster's centroid",
    "The top n words wit...": "The top n words with the lowest cosine distance are examined\n"
    "and the best one is chosen and the cluster's hint",
    "The best hint from ...": "The best hint from all clusters is picked and being hinter\n" "to the gruesser!",
    "Here is a graph of...": "Here is a graph of the guesser's view of a good hinted word",
    "As can be seen2...": "As can be seen, the closest words on board to the hinted word\n"
    "are all from the team's color, while words from other colors\n"
    "are far from the hinted word",
    "With such a hint,": "With such a hint, victory is guaranteed!",
    "Here is a graph of2...": "Here is a graph of the guesser's view of a bad hinted word\n",
    "As can be seen3...": "As can be seen, there is a bad word just as close to the\n"
    "hinted word as the good word, which might confuse the guesser,\n"
    "and lead him to pick up the bad word.",
    "Such a hint will...": "Such a hint will not be chosen.",
}

texts_script = {k: Text(t, font_size=FONT_SIZE_TEXT) for k, t in script.items()}
texts_script["The algorithm uses..."].to_corner(UL)
texts_script["In a nutshell..."].next_to(texts_script["The algorithm uses..."], DOWN)
texts_script["For the sake of..."].to_corner(UL)
texts_script["Here are some..."].to_corner(UL)
texts_script["The word X..."].next_to(texts_script["Here are some..."], DOWN)
texts_script["In each turn..."].to_corner(UL)
texts_script["Two methods..."].next_to(texts_script["In each turn..."], DOWN)
texts_script["In the first cluste..."].to_corner(UL)
texts_script["This graph is divid..."].next_to(texts_script["In the first cluste..."], DOWN)
texts_script["Here is an example..."].next_to(texts_script["This graph is divid..."], DOWN)
texts_script["As can be seen..."].next_to(texts_script["Here is an example..."], DOWN)
texts_script["The second clusteri..."].to_corner(UL)
texts_script["Since there are..."].next_to(texts_script["The second clusteri..."], DOWN)
texts_script["The second task..."].to_corner(UL)
texts_script["In order to find..."].next_to(texts_script["The second task..."], DOWN)
texts_script["An initial centroid..."].next_to(texts_script["In order to find..."], DOWN)
texts_script["Ideally, the centro..."].to_corner(UL)
texts_script["to optimize the..."].next_to(texts_script["Ideally, the centro..."], DOWN)
texts_script["The centroid is the..."].next_to(texts_script["to optimize the..."], DOWN)
texts_script["The attraction forc..."].to_corner(UL)
texts_script["This is done in ord..."].next_to(texts_script["The attraction forc..."], DOWN)
texts_script["After convergence..."].to_corner(UL)
texts_script["The top n words wit..."].next_to(texts_script["After convergence..."], DOWN)
texts_script["The best hint from ..."].next_to(texts_script["The top n words wit..."], DOWN)
texts_script["Here is a graph of..."].to_corner(UL)
texts_script["As can be seen2..."].next_to(texts_script["Here is a graph of..."], DOWN)
texts_script["With such a hint,"].next_to(texts_script["As can be seen2..."], DOWN)
texts_script["Here is a graph of2..."].to_corner(UL)
texts_script["As can be seen3..."].next_to(texts_script["Here is a graph of2..."], DOWN)


class KalirmozExplanation(ThreeDScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.simple_mode = True

    def construct(self):
        theta = 30 * DEGREES
        phi = 75 * DEGREES
        seconds_per_character = 0.02
        axes = ThreeDAxes(x_range=[-2, 2, 1], x_length=4, y_length=4)
        # labels = axes.get_axis_labels(
        #     x_label=Tex("x"), y_label=Tex("y")
        # )
        sphere = Sphere(
            center=(0, 0, 0), radius=SPHERE_RADIUS, resolution=(20, 20), u_range=[0.001, PI - 0.001], v_range=[0, TAU]
        ).set_opacity(0.3)
        arrows_list = [Arrow3D(start=[0, 0, 0], end=vector, thickness=ARROWS_THICKNESS) for vector in vectors_list]
        camera_vectors = [self.coords_to_point(vector * 1.2) for vector in vectors_list]
        words_labels_list = [
            Text(labels_list[i], font_size=FONT_SIZE_LABELS, color=DARK_BROWN).move_to(camera_vectors[i])
            for i in range(list_len)
        ]
        # dots_list = [Dot(point=vector, radius=DOT_SIZE) for vector in vectors_list]
        self.write_3d_text(texts_script["The algorithm uses..."], fade_out=False)
        self.write_3d_text(texts_script["In a nutshell..."], fade_out=False)
        self.wait(3)
        self.remove_3d_text(texts_script["The algorithm uses..."], texts_script["In a nutshell..."])
        self.renderer.camera.light_source.move_to(3 * IN)  # changes the source of the light
        self.set_camera_orientation(phi=phi, theta=theta)  # 75 * DEGREES, theta=30 * DEGREES
        self.add(axes, sphere)
        self.write_3d_text(texts_script["For the sake of..."], fade_out=True)
        # self.play(*[Create(arrows_list[i]) for i in range(list_len)])
        self.add(*arrows_list[0:list_len])
        for i in range(list_len):
            self.add_fixed_in_frame_mobjects(words_labels_list[i])
        words_labels_list[0].add_updater(lambda x: x.move_to(self.coords_to_point(vectors_list[0])))
        words_labels_list[1].add_updater(lambda x: x.move_to(self.coords_to_point(vectors_list[1])))
        words_labels_list[2].add_updater(lambda x: x.move_to(self.coords_to_point(vectors_list[2])))
        words_labels_list[3].add_updater(lambda x: x.move_to(self.coords_to_point(vectors_list[3])))
        words_labels_list[4].add_updater(lambda x: x.move_to(self.coords_to_point(vectors_list[4])))
        words_labels_list[5].add_updater(lambda x: x.move_to(self.coords_to_point(vectors_list[5])))
        words_labels_list[6].add_updater(lambda x: x.move_to(self.coords_to_point(vectors_list[6])))
        self.remove_3d_text(texts_script["For the sake of..."])
        self.write_3d_text(texts_script["Here are some..."])
        self.write_3d_text(texts_script["The word X..."])
        self.remove_3d_text(texts_script["Here are some..."], texts_script["The word X..."])
        self.write_3d_text(texts_script["In each turn..."])
        self.write_3d_text(texts_script["Two methods..."])
        self.remove_3d_text(texts_script["In each turn..."], texts_script["Two methods..."])
        self.write_3d_text(texts_script["In the first cluste..."])
        self.write_3d_text(texts_script["This graph is divid..."])
        self.play(*[Create(connection, run_time=3) for connection in sna_connections_list])
        self.write_3d_text(texts_script["Here is an example..."])
        self.write_3d_text(texts_script["As can be seen..."])
        self.wait(3)
        self.remove_3d_text(
            texts_script["In the first cluste..."],
            texts_script["This graph is divid..."],
            texts_script["Here is an example..."],
            texts_script["As can be seen..."],
        )
        self.write_3d_text(texts_script["The second clusteri..."])
        self.write_3d_text(texts_script["Since there are..."])
        self.animate_random_connections(vectors_list=vectors_list, number_of_examples=15, example_length=0.3)
        self.remove_3d_text(texts_script["The second clusteri..."], texts_script["Since there are..."])
        self.write_3d_text(texts_script["The second task..."])
        self.write_3d_text(texts_script["In order to find..."])
        self.write_3d_text(texts_script["An initial centroid..."])
        self.remove_3d_text(
            texts_script["The second task..."],
            texts_script["In order to find..."],
            texts_script["An initial centroid..."],
        )
        self.write_3d_text(texts_script["Ideally, the centro..."])
        self.write_3d_text(texts_script["to optimize the..."])
        self.write_3d_text(texts_script["The centroid is the..."])
        self.remove_3d_text(
            texts_script["Ideally, the centro..."],
            texts_script["to optimize the..."],
            texts_script["The centroid is the..."],
        )
        self.write_3d_text(texts_script["The attraction forc..."])
        self.write_3d_text(texts_script["This is done in ord..."])
        self.remove_3d_text(texts_script["The attraction forc..."], texts_script["This is done in ord..."])
        self.write_3d_text(texts_script["After convergence..."])
        self.write_3d_text(texts_script["The top n words wit..."])
        self.write_3d_text(texts_script["The best hint from ..."])
        self.remove_3d_text(
            texts_script["After convergence..."],
            texts_script["The top n words wit..."],
            texts_script["The best hint from ..."],
        )
        self.write_3d_text(texts_script["Here is a graph of..."])
        self.write_3d_text(texts_script["As can be seen2..."])
        self.write_3d_text(texts_script["Such a hint will..."])
        self.remove_3d_text(
            texts_script["Here is a graph of..."],
            texts_script["As can be seen2..."],
            texts_script["Such a hint will..."],
        )
        self.write_3d_text(texts_script["Here is a graph of2..."])
        self.write_3d_text(texts_script["As can be seen3..."])
        self.animate_random_connections(vectors_list, 10, 0.3)

    def animate_physical_system(self, starting_point: np.array,
                                attractors: List[np.array,...], repellers=List[np.array,...]):
        trajectory = record_trajectory(starting_point=starting_point, attractors=attractors, repellers=repellers)
        centroid = Arrow3D(start=[0,0,0], end=starting_point)
        for i in range(len(trajectory)):
            temp_centroid = Arrow3D(start=[0,0,0], end=trajectory[i])
            self.play(Transform(centroid, temp_centroid))


    def animate_random_connections(self, vectors_list, number_of_examples, example_length):
        for i in range(number_of_examples):
            connections = generate_random_connections(vectors_list, average_cluster_size=3)
            self.add(*connections)
            self.wait(example_length)
            self.remove(*connections)

    def coords_to_point(self, coords):
        theta = -self.camera.get_theta()
        phi = -self.camera.get_phi()
        coords_np = np.array(coords)
        constant_rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        theta_rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        phi_rotation = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
        rotated = phi_rotation @ theta_rotation @ constant_rotation @ coords_np
        return rotated

    def update_position_to_camera(self, mob, coordinate):
        mob.move_to(self.coords_to_point(coordinate))

    def write_3d_text(self, text_object, fade_out=False, waiting_time=None):
        if waiting_time is None:
            waiting_time = text_len_to_time(text_object.original_text)
        self.add_fixed_in_frame_mobjects(text_object)
        if self.simple_mode:
            self.add(text_object)
            if fade_out:
                self.wait(waiting_time)
                self.remove(text_object)
        else:
            self.play(Write(text_object))
            if fade_out:
                self.wait(waiting_time)
                self.play(FadeOut(text_object))

    def remove_3d_text(self, *text_objects):
        if self.simple_mode:
            self.remove(*text_objects)
        else:
            self.play([FadeOut(text_object) for text_object in text_objects])


class Intro(Scene):
    def construct(self):
        t1 = Text("Code Names Algorithm", color=BLUE)
        t2 = Text("by the Kali brothers", color=RED).scale(0.8).next_to(t1, DOWN)
        self.play(Write(t1))
        self.wait()
        self.play(Write(t2))
        self.wait()
        self.remove(t1, t2)
        self.wait()

from manim import *
import numpy as np
import random
from typing import Callable, List, Optional
from codenames.solvers.utils.algebra import geodesic, cosine_distance, normalize
from codenames.solvers.sna_solvers.sna_hinter import step_from_forces, ForceNode  # , opponent_force, friendly_force
from scipy.interpolate import interp1d


def random_ints(n: int, k: int) -> List[int]:
    ints_list = [random.randint(0, n) for i in range(k)]
    return ints_list


def geodesic_object(v, u):
    geodesic_function = geodesic(v, u)
    return ParametricFunction(geodesic_function, t_range=[0, 1])


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
            geodesic_function = geodesic_object(cluster_vectors[i], cluster_vectors[j])
            connections_list.append(geodesic_function)
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


def enrich_nodes(centroid, nodes_list):  #:List[ForceNode,...]
    nodes = []
    for node in nodes_list:
        d = cosine_distance(centroid, node.force_origin)
        if node.force_sign:
            node.force_size = attractor_force(d)
        else:
            node.force_size = repeller_force(d)
        nodes.append(node)
    return nodes


def record_trajectory(starting_point, nodes_list, num_of_iterations=50, arc_radians=0.01):
    trajectory = [starting_point]
    centroid = starting_point
    for i in range(num_of_iterations):
        nodes = enrich_nodes(centroid, nodes_list)
        centroid = step_from_forces(centroid, nodes, arc_radians=arc_radians)
        trajectory.append(centroid)
    return trajectory


def attractor_force(d):
    return -d


def repeller_force(d):
    return 0.8 * (1 / (d + 1) - 0.5)


# def connecting_line(v, u):
#   c = v.T @ u
#   f = lambda t: (t*v+(1-t)*u) / np.sqrt(t**2+(1-t**2) + 2*t*(1-t)*c)
#   return f

SPHERE_RADIUS = 1
FONT_SIZE_LABELS = 12
FONT_SIZE_TEXT = 30
ARROWS_THICKNESS = 0.001
DOT_SIZE = 0.2
LABELS_COLOR = GREEN_E
SKI_VEC = polar_to_cartesian(SPHERE_RADIUS, 0.5 * PI, 0)
WATER_VEC = polar_to_cartesian(SPHERE_RADIUS, 0.5 * PI, 0.2 * PI)
BEACH_VEC = polar_to_cartesian(SPHERE_RADIUS, 0.70 * PI, 0.27 * PI)
PARK_VEC = polar_to_cartesian(SPHERE_RADIUS, 0.75 * PI, 0.1 * PI)
JUPITER_VEC = polar_to_cartesian(SPHERE_RADIUS, 0.2 * PI, -0.3 * PI)
NEWTON_VEC = polar_to_cartesian(SPHERE_RADIUS, 0.27 * PI, -0.11 * PI)
TEACHER_VEC = polar_to_cartesian(SPHERE_RADIUS, 0.3 * PI, -0.2 * PI)

vectors_list = [SKI_VEC, WATER_VEC, BEACH_VEC, PARK_VEC, JUPITER_VEC, NEWTON_VEC, TEACHER_VEC]

generate_random_connections(vectors_list)


labels_list = ["ski", "water", "beach", "park", "jupiter", "newton", "teacher"]
words_list_len = len(vectors_list)

sna_connections_list = [
    ParametricFunction(geodesic(SKI_VEC, WATER_VEC), t_range=[0, 1]),
    ParametricFunction(geodesic(BEACH_VEC, PARK_VEC), t_range=[0, 1]),
    ParametricFunction(geodesic(JUPITER_VEC, NEWTON_VEC), t_range=[0, 1]),
    ParametricFunction(geodesic(TEACHER_VEC, JUPITER_VEC), t_range=[0, 1]),
    ParametricFunction(geodesic(NEWTON_VEC, TEACHER_VEC), t_range=[0, 1]),
]

script_dict = {
    "The algorithm uses...": "The Kalirmoz algorithm uses a Word2Vec model for the linguistic knowledge",
    "In a nutshell...": "In a nutshell, the Word2Vec assign each word with an n-dimensional\n"
    "vector (usually n=50, 100, 300), in a way such that words that\n"
    "tent to appear in the same context have small angle between them",
    "For the sake of...": "For the sake of this video, we will represent the words vectors\n"
    "as 3-dimensional vectors",
    "Here are some...": "Here are some words and their corresponding vectors.",
    "The word X...": 'The word "water" is close to the words "ski" and "beach" and far\n'
    "from the word newton as indeed semantically, the words water,ski and beach all appear in the\n"
    "same contexts, while the word newton usually appears in other contexts",
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

scr = {k: Text(t, font_size=FONT_SIZE_TEXT) for k, t in script_dict.items()}
scr["The algorithm uses..."].to_corner(UL)
scr["In a nutshell..."].next_to(scr["The algorithm uses..."], DOWN).align_to(scr["The algorithm uses..."], LEFT)
scr["For the sake of..."].to_corner(UL)
scr["Here are some..."].to_corner(UL)
scr["The word X..."].next_to(scr["Here are some..."], DOWN).align_to(scr["Here are some..."], LEFT)
scr["In each turn..."].to_corner(UL)
scr["Two methods..."].next_to(scr["In each turn..."], DOWN).align_to(scr["In each turn..."], LEFT)
scr["In the first cluste..."].to_corner(UL)
scr["This graph is divid..."].next_to(scr["In the first cluste..."], DOWN).align_to(scr["In the first cluste..."], LEFT)
scr["Here is an example..."].next_to(scr["This graph is divid..."], DOWN).align_to(scr["This graph is divid..."], LEFT)
scr["As can be seen..."].next_to(scr["Here is an example..."], DOWN).align_to(scr["Here is an example..."], LEFT)
scr["The second clusteri..."].to_corner(UL)
scr["Since there are..."].next_to(scr["The second clusteri..."], DOWN).align_to(scr["The second clusteri..."], LEFT)
scr["The second task..."].to_corner(UL)
scr["In order to find..."].next_to(scr["The second task..."], DOWN).align_to(scr["The second task..."], LEFT)
scr["An initial centroid..."].next_to(scr["In order to find..."], DOWN).align_to(scr["In order to find..."], LEFT)
scr["Ideally, the centro..."].to_corner(UL)
scr["to optimize the..."].next_to(scr["Ideally, the centro..."], DOWN).align_to(scr["Ideally, the centro..."], LEFT)
scr["The centroid is the..."].next_to(scr["to optimize the..."], DOWN).align_to(scr["to optimize the..."], LEFT)
scr["The attraction forc..."].to_corner(UL)
scr["This is done in ord..."].next_to(scr["The attraction forc..."], DOWN).align_to(scr["The attraction forc..."], LEFT)
scr["After convergence..."].to_corner(UL)
scr["The top n words wit..."].next_to(scr["After convergence..."], DOWN).align_to(scr["After convergence..."], LEFT)
scr["The best hint from ..."].next_to(scr["The top n words wit..."], DOWN).align_to(scr["The top n words wit..."], LEFT)
scr["Here is a graph of..."].to_corner(UL)
scr["As can be seen2..."].next_to(scr["Here is a graph of..."], DOWN).align_to(scr["Here is a graph of..."], LEFT)
scr["With such a hint,"].next_to(scr["As can be seen2..."], DOWN).align_to(scr["As can be seen2..."], LEFT)
scr["Here is a graph of2..."].to_corner(UL)
scr["As can be seen3..."].next_to(scr["Here is a graph of2..."], DOWN).align_to(scr["Here is a graph of2..."], LEFT)


class KalirmozExplanation(ThreeDScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.simple_mode = False

    def construct(self):

        theta = 30 * DEGREES
        phi = 75 * DEGREES
        seconds_per_character = 0.02
        axes = ThreeDAxes(
            x_range=[-1.5, 1.5, 1], y_range=[-1.5, 1.5, 1], z_range=[-1.5, 1.5, 1], x_length=8, y_length=8, z_length=8
        )
        sphere = Sphere(
            center=(0, 0, 0), radius=SPHERE_RADIUS, resolution=(20, 20), u_range=[0.001, PI - 0.001], v_range=[0, TAU]
        ).set_opacity(0.3)

        text_box = Rectangle(color=DARK_BROWN, fill_color=BLACK, fill_opacity=1, height=7.0, width=5.0).to_edge(LEFT).shift(0.2*LEFT)
        self.add_fixed_in_frame_mobjects(text_box)
        self.add(text_box)
        arrows_list = [Line(start=[0, 0, 0], end=vector) for vector in vectors_list]  # stroke_width=ARROWS_THICKNESS
        camera_vectors = [self.coords_to_point(vector * 1.2) for vector in vectors_list]
        words_labels_list = [
            Text(labels_list[i], font_size=FONT_SIZE_LABELS, color=LABELS_COLOR).move_to(camera_vectors[i])
            for i in range(words_list_len)
        ]

        # self.write_3d_text(scr["The algorithm uses..."], fade_out=False)
        # self.write_3d_text(scr["In a nutshell..."], fade_out=False)
        # self.remove_3d_text(scr["The algorithm uses..."], scr["In a nutshell..."])
        self.renderer.camera.light_source.move_to(3 * IN)
        self.set_camera_orientation(phi=phi, theta=theta)  # 75 * DEGREES, theta=30 * DEGREES
        self.move_camera(frame_center=np.array([-2, -2, 1]))
        self.add(axes, sphere)
        # self.write_3d_text(scr["For the sake of..."], fade_out=True)
        for i in range(words_list_len):
            self.add_fixed_orientation_mobjects(words_labels_list[i])
            # words_labels_list[i].add_updater(lambda x, i=i: x.move_to(self.coords_to_point(vectors_list[i])))
        self.add(*words_labels_list)
        self.add(*arrows_list[0:words_list_len])
        self.begin_ambient_camera_rotation()
        self.wait(2)
        # colors = [RED, GREEN, BLUE, YELLOW, ORANGE]
        # values = [0.5, 0.6, 0.9, 0.45, 0.96, 0.73, 0.2, 0.4, 0.49, 0.75, 0.9]
        # bar_chart = BarChart(values, bar_colors=colors)
        # self.play(DrawBorderThenFill(bar_chart, run_time=25))
        # self.wait(3)

        # self.begin_ambient_camera_rotation(rate=0.1)
        # self.remove_3d_text(scr["For the sake of..."])
        # self.write_3d_text(scr["Here are some..."])
        # self.play(*[Create(arrows_list[i]) for i in range(words_list_len)])
        # # self.add(*arrows_list[0:words_list_len])

        # self.write_3d_text(scr["The word X..."])
        # self.remove_3d_text(scr["Here are some..."], scr["The word X..."])
        # self.write_3d_text(scr["In each turn..."])
        # self.write_3d_text(scr["Two methods..."])
        # self.remove_3d_text(scr["In each turn..."], scr["Two methods..."])
        # self.write_3d_text(scr["In the first cluste..."])
        # self.write_3d_text(scr["This graph is divid..."])
        # self.play(*[Create(connection, run_time=3) for connection in sna_connections_list])
        # self.write_3d_text(scr["Here is an example..."])
        # self.write_3d_text(scr["As can be seen..."])
        # self.wait(3)
        # self.remove_3d_text(
        #     scr["In the first cluste..."],
        #     scr["This graph is divid..."],
        #     scr["Here is an example..."],
        #     scr["As can be seen..."],
        # )
        # self.write_3d_text(scr["The second clusteri..."])
        # self.write_3d_text(scr["Since there are..."])
        # self.animate_random_connections(vectors_list=vectors_list, number_of_examples=15, example_length=0.3)
        # self.remove_3d_text(scr["The second clusteri..."], scr["Since there are..."])
        # self.write_3d_text(scr["The second task..."])
        # self.write_3d_text(scr["In order to find..."])
        # self.write_3d_text(scr["An initial centroid..."])
        # self.remove_3d_text(
        #     scr["The second task..."],
        #     scr["In order to find..."],
        #     scr["An initial centroid..."],
        # )
        # self.write_3d_text(scr["Ideally, the centro..."])
        # self.write_3d_text(scr["to optimize the..."])
        # self.write_3d_text(scr["The centroid is the..."])
        #
        # starting_point = polar_to_cartesian(1, 0.52 * PI, 1.95 * PI)
        # nodes_list = [ForceNode(SKI_VEC, True), ForceNode(WATER_VEC, True), ForceNode(PARK_VEC, False)]
        # self.animate_physical_system(
        #     starting_point=starting_point, nodes_list=nodes_list, num_of_iterations=1500, arc_radians=0.01
        # )
        #
        # self.remove_3d_text(
        #     scr["Ideally, the centro..."],
        #     scr["to optimize the..."],
        #     scr["The centroid is the..."],
        # )
        # self.write_3d_text(scr["The attraction forc..."])
        # self.write_3d_text(scr["This is done in ord..."])
        # self.remove_3d_text(scr["The attraction forc..."], scr["This is done in ord..."])
        # self.write_3d_text(scr["After convergence..."])
        # self.write_3d_text(scr["The top n words wit..."])
        # self.write_3d_text(scr["The best hint from ..."])
        # self.remove_3d_text(
        #     scr["After convergence..."],
        #     scr["The top n words wit..."],
        #     scr["The best hint from ..."],
        # )
        # self.write_3d_text(scr["Here is a graph of..."])
        # self.write_3d_text(scr["As can be seen2..."])
        # self.write_3d_text(scr["Such a hint will..."])
        # self.remove_3d_text(
        #     scr["Here is a graph of..."],
        #     scr["As can be seen2..."],
        #     scr["Such a hint will..."],
        # )
        # self.write_3d_text(scr["Here is a graph of2..."])
        # self.write_3d_text(scr["As can be seen3..."])
        # self.animate_random_connections(vectors_list, 10, 0.3)

    def animate_physical_system(
        self, starting_point: np.array, nodes_list, num_of_iterations=10, arc_radians=0.01
    ):  #:List[np.array,...]
        trajectory = record_trajectory(
            starting_point=starting_point,
            nodes_list=nodes_list,
            num_of_iterations=num_of_iterations,
            arc_radians=arc_radians,
        )
        x = np.linspace(0, 1, len(trajectory))
        y = np.stack(trajectory)
        trajectory_interp = interp1d(x, y, axis=0)
        t = ValueTracker(0)
        centroid = Line(start=[0, 0, 0], end=starting_point)
        centroid.add_updater(lambda x: x.become(Line(start=[0, 0, 0], end=normalize(trajectory_interp(t.get_value())))))
        # forces = []
        # for i, node in enumerate(nodes_list):
        #     force = geodesic_object(node.force_origin, normalize(centroid.get_end()))
        #     if node.force_sign:
        #         force.set_color(BLUE)
        #     else:
        #         force.set_color(RED)
        #     force.add_updater(lambda x, i=i: x.become(geodesic_object(node.force_origin, normalize(trajectory_interp(t.get_value())))))
        #     forces.append(force)
        park_force = geodesic_object(PARK_VEC, normalize(centroid.get_end())).set_color(RED)
        park_force.add_updater(
            lambda x: x.become(geodesic_object(PARK_VEC, normalize(trajectory_interp(t.get_value())))).set_color(RED)
        )
        ski_force = geodesic_object(SKI_VEC, normalize(centroid.get_end())).set_color(BLUE)
        ski_force.add_updater(
            lambda x: x.become(geodesic_object(SKI_VEC, normalize(trajectory_interp(t.get_value())))).set_color(BLUE)
        )
        water_force = geodesic_object(WATER_VEC, normalize(centroid.get_end())).set_color(BLUE)
        water_force.add_updater(
            lambda x: x.become(geodesic_object(WATER_VEC, normalize(trajectory_interp(t.get_value())))).set_color(BLUE)
        )
        self.play(Create(centroid), Create(park_force), Create(ski_force), Create(water_force))
        self.play(t.animate.set_value(1), run_time=4, rate_func=linear)

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
            self.wait(waiting_time)
            if fade_out:
                self.remove(text_object)
        else:
            self.play(Write(text_object))
            self.wait(waiting_time)
            if fade_out:
                self.play(FadeOut(text_object))

    def remove_3d_text(self, *text_objects):
        if self.simple_mode:
            self.remove(*text_objects)
        else:
            self.play(*[FadeOut(text_object) for text_object in text_objects])


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


# test_scene = KalirmozExplanation()
# starting_point = polar_to_cartesian(1, 0.52 * PI, 1.95 * PI)
# nodes_list = [ForceNode(SKI_VEC, True), ForceNode(WATER_VEC, True), ForceNode(PARK_VEC, False)]
# test_scene.animate_physical_system(
#     starting_point=starting_point, nodes_list=nodes_list, num_of_iterations=1500, arc_radians=0.01
# )

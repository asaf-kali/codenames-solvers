# flake8: noqa: E405
# type: ignore

import random
from typing import List

import pandas as pd
from manim import *
from scipy.interpolate import interp1d

from codenames.solvers.sna_solvers.sna_hinter import step_from_forces, ForceNode
from codenames.solvers.utils.algebra import geodesic, cosine_distance, normalize


def text2color(text):
    if text == "blue":
        return BLUE
    if text == "red":
        return RED
    if text == "black":
        return BLACK
    if text == "gray":
        return GRAY


def random_ints(n: int, k: int) -> List[int]:
    ints_list = [random.randint(0, n) for i in range(k)]
    return ints_list


def geodesic_object(v, u):
    geodesic_function = geodesic(v, u)
    return ParametricFunction(geodesic_function, t_range=[0, 1], color=ORANGE)


def generate_random_subsets(elements_list: List, average_subset_size: float) -> List[List]:
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


def generate_random_connections(vectors_list, average_cluster_size):
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


def text_len_to_time(text, min_time=2, seconds_per_char=0.05):
    n = len(text)
    return np.max([min_time, n * seconds_per_char])


def enrich_nodes(centroid, nodes_list):  # :List[ForceNode,...]
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
FONT_SIZE_LABELS = 14
FONT_SIZE_TEXT = 25
ARROWS_THICKNESS = 0.001
DOT_SIZE = 0.2
LABELS_COLOR = PURE_RED
SKI_VEC = polar_to_cartesian(SPHERE_RADIUS, 0.5 * PI, 0)
WATER_VEC = polar_to_cartesian(SPHERE_RADIUS, 0.5 * PI, 0.2 * PI)
BEACH_VEC = polar_to_cartesian(SPHERE_RADIUS, 0.70 * PI, 0.27 * PI)
PARK_VEC = polar_to_cartesian(SPHERE_RADIUS, 0.75 * PI, 0.1 * PI)
JUPITER_VEC = polar_to_cartesian(SPHERE_RADIUS, 0.2 * PI, -0.3 * PI)
NEWTON_VEC = polar_to_cartesian(SPHERE_RADIUS, 0.27 * PI, -0.11 * PI)
TEACHER_VEC = polar_to_cartesian(SPHERE_RADIUS, 0.35 * PI, -0.2 * PI)

vectors_list = [SKI_VEC, WATER_VEC, BEACH_VEC, PARK_VEC, JUPITER_VEC, NEWTON_VEC, TEACHER_VEC]
labels_list = ["ski", "water", "beach", "park", "jupiter", "newton", "teacher"]
words_list_len = len(vectors_list)

sna_connections_list = [
    ParametricFunction(geodesic(SKI_VEC, WATER_VEC), t_range=[0, 1]),
    ParametricFunction(geodesic(BEACH_VEC, PARK_VEC), t_range=[0, 1]),
    ParametricFunction(geodesic(JUPITER_VEC, NEWTON_VEC), t_range=[0, 1]),
    ParametricFunction(geodesic(TEACHER_VEC, JUPITER_VEC), t_range=[0, 1]),
    ParametricFunction(geodesic(NEWTON_VEC, TEACHER_VEC), t_range=[0, 1]),
]


class Scr:
    THE_ALGORITHM_USES = Text(
        "The Kalirmoz algorithm uses a Word2Vec model for the linguistic knowledge.", font_size=FONT_SIZE_TEXT
    )
    IN_A_NUTSHELL = Text(
        """In a nutshell, Word2Vec assigns each word an n-dimensional
vector (usually n=50, 100, 300), in a way such that words that
tend to appear in the same context have small angle between them.""",
        font_size=FONT_SIZE_TEXT,
    )
    FOR_THE_SAKE_OF = Text(
        """For demonstration purposes, we will represent these vectors
in 3-dimensional space.""",
        font_size=FONT_SIZE_TEXT,
    )

    HERE_ARE_SOME = Text(
        """Here are some words and their
corresponding vectors:""",
        font_size=FONT_SIZE_TEXT,
    )
    THE_WORD_X = Text(
        """The word "water" is close to
"ski" and "beach"
and far from "newton",
as we would semantically
expect.""",
        font_size=FONT_SIZE_TEXT,
    )
    IN_EACH_TURN = Text(
        """In each turn, the first task
of the hinter is to find a
proper subset of words
(usually 2 to 4 words),
on which to hint.""",
        font_size=FONT_SIZE_TEXT,
    )
    TWO_METHODS = Text("""Two methods of clustering were implemented.""", font_size=FONT_SIZE_TEXT)
    FIRST_CLUSTERING_METHOD = Text(
        """In the first clustering
method, the words are
treated as nodes in a
graph, with edges
weights correlated to
their cosine similarity.""",
        font_size=FONT_SIZE_TEXT,
    )
    THIS_GRAPH_IS_DIVIDED = Text(
        """This graph is divided into
communities using the louvain
SNA algorithm, and each
community is taken as an
optional cluster of words to
hint about.""",
        font_size=FONT_SIZE_TEXT,
    )
    HERE_IS_AN_EXAMPLE = Text(
        """Here is an example of 25 words and their louvain clustering result:""", font_size=FONT_SIZE_TEXT
    )
    AS_CAN_BE_SEEN = Text(
        """As can be seen, semantically
close words are put within the
same cluster.""",
        font_size=FONT_SIZE_TEXT,
    )
    THE_SECOND_CLUSTERING = Text("""The second clustering method is much simpler:""", font_size=FONT_SIZE_TEXT)
    SINCE_THERE_ARE = Text(
        """Since there are at most 9 cards
to hint about, it is feasible
to just iterate over all possible
subsets and choose the best
one.""",
        font_size=FONT_SIZE_TEXT,
    )
    THE_SECOND_TASK = Text(
        """The second task of the hinter is to choose a hinting word for the cluster.""", font_size=FONT_SIZE_TEXT
    )
    IN_ORDER_TO_FIND = Text(
        """In order to find a hinting word
for a cluster, the hinter
generates a "centroid" vector
for the cluster, to search real
words near by.""",
        font_size=FONT_SIZE_TEXT,
    )
    AN_INITIAL_CENTROID = Text(
        """An initial "centroid" is
proposed as the Center of Mass
of the cluster's vectors""",
        font_size=FONT_SIZE_TEXT,
    )
    IDEALLY_THE_CENTROID = Text(
        """Ideally, the centroid would be
close to all the cluster's
words and far from words of
other colors. (where "close"
and "far") are considered in
the cosine distance metric.""",
        font_size=FONT_SIZE_TEXT,
    )
    TO_OPTIMIZE_THE = Text(
        """To optimize the centroid, the
words in the board (from
all colors) are considered
as a physical system, where
every vector from the color
of the hinter is an attractor,
and every word from other
color is a repeller.""",
        font_size=FONT_SIZE_TEXT,
    )
    THE_CENTROID_IS_THE = Text(
        """The centroid is then being
pushed and pulled by the words
of the board until converging
to a point where it is both
far away from bad words, and
still close to the cluster
words.""",
        font_size=FONT_SIZE_TEXT,
    )
    THE_ATTRACTION_FORCE = Text(
        """The attraction force acts like
a spring, where if the centroid
is to far, the spring can be
"torn" apart and is no longer
considered part of the cluster.""",
        font_size=FONT_SIZE_TEXT,
    )
    THIS_IS_DONE_IN_ORD = Text(
        """This is done in order to allow
outliers in the cluster to be
neglected.""",
        font_size=FONT_SIZE_TEXT,
    )
    AFTER_CONVERGENCE = Text(
        """After convergence, all there
needs to be done is to pick up a
word near-by the optimized
cluster's centroid""",
        font_size=FONT_SIZE_TEXT,
    )
    THE_TOP_N_WORDS_WIT = Text(
        """The top n-words with the lowest
cosine distance are examined
and the best one is chosen as
the cluster's hint""",
        font_size=FONT_SIZE_TEXT,
    )
    THE_BEST_HINT_FROM = Text(
        """The best hint from all clusters
is picked and sent
to the guesser!""",
        font_size=FONT_SIZE_TEXT,
    )
    HERE_IS_A_GRAPH_OF = Text(
        """Here is a graph of the
guesser's view of a good
hinted word.""",
        font_size=FONT_SIZE_TEXT,
    )
    AS_CAN_BE_SEEN2 = Text(
        """As can be seen, the closest
words on board to the hinted
word are all from the team's
color, while words from other
colors are far from the hinted
word.""",
        font_size=FONT_SIZE_TEXT,
    )
    WITH_SUCH_A_HINT = Text(
        """With such a hint, victory is
guaranteed!""",
        font_size=FONT_SIZE_TEXT,
    )
    HERE_IS_A_GRAPH_OF2 = Text(
        """Here is a graph of the
guesser's view of a bad hinted
word""",
        font_size=FONT_SIZE_TEXT,
    )
    AS_CAN_BE_SEEN3 = Text(
        """As can be seen, there is a bad
word just as close to the
hinted word as the good word,
which might confuse the guesser,
and lead him to pick up the bad
word.""",
        font_size=FONT_SIZE_TEXT,
    )
    SUCH_A_HINT_WILL = Text("""Such a hint will not be chosen.""", font_size=FONT_SIZE_TEXT)


# scr = {k: Text(t, font_size=FONT_SIZE_TEXT) for k, t in script_dict.items()}
Scr.THE_ALGORITHM_USES.to_corner(UL)
Scr.IN_A_NUTSHELL.next_to(Scr.THE_ALGORITHM_USES, DOWN).align_to(Scr.THE_ALGORITHM_USES, LEFT)
Scr.FOR_THE_SAKE_OF.to_corner(UL)
Scr.HERE_ARE_SOME.to_corner(UL)
Scr.THE_WORD_X.next_to(Scr.HERE_ARE_SOME, DOWN).align_to(Scr.HERE_ARE_SOME, LEFT)
Scr.IN_EACH_TURN.to_corner(UL)
Scr.TWO_METHODS.next_to(Scr.IN_EACH_TURN, DOWN).align_to(Scr.IN_EACH_TURN, LEFT)
Scr.FIRST_CLUSTERING_METHOD.to_corner(UL)
Scr.THIS_GRAPH_IS_DIVIDED.next_to(Scr.FIRST_CLUSTERING_METHOD, DOWN).align_to(Scr.FIRST_CLUSTERING_METHOD, LEFT)
Scr.HERE_IS_AN_EXAMPLE.next_to(Scr.THIS_GRAPH_IS_DIVIDED, DOWN).align_to(Scr.THIS_GRAPH_IS_DIVIDED, LEFT)
Scr.AS_CAN_BE_SEEN.next_to(Scr.HERE_IS_AN_EXAMPLE, DOWN).align_to(Scr.HERE_IS_AN_EXAMPLE, LEFT)
Scr.THE_SECOND_CLUSTERING.to_corner(UL)
Scr.SINCE_THERE_ARE.next_to(Scr.THE_SECOND_CLUSTERING, DOWN).align_to(Scr.THE_SECOND_CLUSTERING, LEFT)
Scr.THE_SECOND_TASK.to_corner(UL)
Scr.IN_ORDER_TO_FIND.next_to(Scr.THE_SECOND_TASK, DOWN).align_to(Scr.THE_SECOND_TASK, LEFT)
Scr.AN_INITIAL_CENTROID.next_to(Scr.IN_ORDER_TO_FIND, DOWN).align_to(Scr.IN_ORDER_TO_FIND, LEFT)
Scr.IDEALLY_THE_CENTROID.to_corner(UL)
Scr.TO_OPTIMIZE_THE.next_to(Scr.IDEALLY_THE_CENTROID, DOWN).align_to(Scr.IDEALLY_THE_CENTROID, LEFT)
Scr.THE_CENTROID_IS_THE.next_to(Scr.TO_OPTIMIZE_THE, DOWN).align_to(Scr.TO_OPTIMIZE_THE, LEFT)
Scr.THE_ATTRACTION_FORCE.to_corner(UL)
Scr.THIS_IS_DONE_IN_ORD.next_to(Scr.THE_ATTRACTION_FORCE, DOWN).align_to(Scr.THE_ATTRACTION_FORCE, LEFT)
Scr.AFTER_CONVERGENCE.to_corner(UL)
Scr.THE_TOP_N_WORDS_WIT.next_to(Scr.AFTER_CONVERGENCE, DOWN).align_to(Scr.AFTER_CONVERGENCE, LEFT)
Scr.THE_BEST_HINT_FROM.next_to(Scr.THE_TOP_N_WORDS_WIT, DOWN).align_to(Scr.THE_TOP_N_WORDS_WIT, LEFT)
Scr.HERE_IS_A_GRAPH_OF.to_corner(UL).scale(0.7)
Scr.AS_CAN_BE_SEEN2.next_to(Scr.HERE_IS_A_GRAPH_OF, DOWN).align_to(Scr.HERE_IS_A_GRAPH_OF, LEFT).scale(0.7)
Scr.WITH_SUCH_A_HINT.next_to(Scr.AS_CAN_BE_SEEN2, DOWN).align_to(Scr.AS_CAN_BE_SEEN2, LEFT).scale(0.7)
Scr.HERE_IS_A_GRAPH_OF2.to_corner(UL).scale(0.7)
Scr.AS_CAN_BE_SEEN3.next_to(Scr.HERE_IS_A_GRAPH_OF2, DOWN).align_to(Scr.HERE_IS_A_GRAPH_OF2, LEFT).scale(0.7)
Scr.SUCH_A_HINT_WILL.to_corner(UL).scale(0.7)


class KalirmozExplanation(ThreeDScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.simple_mode = False

    def construct(self):

        t1 = Text("Code Names Algorithm", color=BLUE)
        t2 = Text("by the Kali brothers", color=RED).scale(0.8).next_to(t1, DOWN)
        self.write_3d_text(t1)
        self.write_3d_text(t2)
        self.play(FadeOut(t1), FadeOut(t2))
        self.write_3d_text(Scr.THE_ALGORITHM_USES, fade_out=False)
        self.write_3d_text(Scr.IN_A_NUTSHELL, fade_out=False)
        self.remove_3d_text(Scr.THE_ALGORITHM_USES, Scr.IN_A_NUTSHELL)
        self.write_3d_text(Scr.FOR_THE_SAKE_OF, fade_out=True)
        self.remove_3d_text(Scr.FOR_THE_SAKE_OF)

        theta = 30 * DEGREES
        phi = 75 * DEGREES
        axes = ThreeDAxes(
            x_range=[-1.5, 1.5, 1], y_range=[-1.5, 1.5, 1], z_range=[-1.5, 1.5, 1], x_length=8, y_length=8, z_length=8
        )
        sphere = Sphere(
            center=(0, 0, 0), radius=SPHERE_RADIUS, resolution=(20, 20), u_range=[0.001, PI - 0.001], v_range=[0, TAU]
        ).set_opacity(0.3)

        text_box = (
            Rectangle(color=DARK_BROWN, fill_color=BLACK, fill_opacity=1, height=7.1, width=5.1)
            .to_edge(LEFT)
            .shift(0.2 * LEFT)
        )
        self.add_fixed_in_frame_mobjects(text_box)
        self.add(text_box)
        self.renderer.camera.light_source.move_to(3 * IN)
        self.set_camera_orientation(phi=phi, theta=theta)  # 75 * DEGREES, theta=30 * DEGREES
        self.begin_ambient_camera_rotation(rate=0.1)
        self.play(Create(axes), Create(sphere))
        self.wait(1)
        self.write_3d_text(Scr.HERE_ARE_SOME)
        # camera_vectors = [self.coords_to_point(vector * 1.2) for vector in vectors_list]
        words_labels_list = [
            Text(labels_list[i], font_size=FONT_SIZE_LABELS, color=LABELS_COLOR).move_to(vectors_list[i])
            for i in range(words_list_len)
        ]
        arrows_list = [Line(start=[0, 0, 0], end=vector) for vector in vectors_list]  # stroke_width=ARROWS_THICKNESS
        for i in range(words_list_len):
            self.add_fixed_orientation_mobjects(words_labels_list[i])
            # words_labels_list[i].add_updater(lambda x, i=i: x.move_to(self.coords_to_point(vectors_list[i])))
        self.play(
            *[Create(arrows_list[i]) for i in range(words_list_len)],
            *[FadeIn(words_labels_list[i]) for i in range(words_list_len)],
        )
        self.write_3d_text(Scr.THE_WORD_X)
        self.remove_3d_text(Scr.HERE_ARE_SOME, Scr.THE_WORD_X)
        self.write_3d_text(Scr.IN_EACH_TURN)
        self.write_3d_text(Scr.TWO_METHODS)
        self.remove_3d_text(Scr.IN_EACH_TURN, Scr.TWO_METHODS)
        self.write_3d_text(Scr.FIRST_CLUSTERING_METHOD)
        self.write_3d_text(Scr.THIS_GRAPH_IS_DIVIDED)
        self.play(*[Create(connection, run_time=3) for connection in sna_connections_list])
        self.write_3d_text(Scr.HERE_IS_AN_EXAMPLE)
        self.write_3d_text(Scr.AS_CAN_BE_SEEN)
        self.wait(3)
        self.remove_3d_text(
            Scr.FIRST_CLUSTERING_METHOD,
            Scr.THIS_GRAPH_IS_DIVIDED,
            Scr.HERE_IS_AN_EXAMPLE,
            Scr.AS_CAN_BE_SEEN,
        )
        self.write_3d_text(Scr.THE_SECOND_CLUSTERING)
        self.write_3d_text(Scr.SINCE_THERE_ARE)
        self.animate_random_connections(vectors_list=vectors_list, number_of_examples=25, example_length=0.3)
        self.remove_3d_text(Scr.THE_SECOND_CLUSTERING, Scr.SINCE_THERE_ARE)
        self.write_3d_text(Scr.THE_SECOND_TASK)
        self.write_3d_text(Scr.IN_ORDER_TO_FIND)
        self.write_3d_text(Scr.AN_INITIAL_CENTROID)
        self.remove_3d_text(
            Scr.THE_SECOND_TASK,
            Scr.IN_ORDER_TO_FIND,
            Scr.AN_INITIAL_CENTROID,
        )
        self.write_3d_text(Scr.IDEALLY_THE_CENTROID)
        self.write_3d_text(Scr.TO_OPTIMIZE_THE)
        self.write_3d_text(Scr.THE_CENTROID_IS_THE)

        starting_point = polar_to_cartesian(1, 0.52 * PI, 1.95 * PI)
        nodes_list = [ForceNode(SKI_VEC, True), ForceNode(WATER_VEC, True), ForceNode(PARK_VEC, False)]
        self.animate_physical_system(
            starting_point=starting_point, nodes_list=nodes_list, num_of_iterations=1500, arc_radians=0.01
        )

        self.remove_3d_text(
            Scr.IDEALLY_THE_CENTROID,
            Scr.TO_OPTIMIZE_THE,
            Scr.THE_CENTROID_IS_THE,
        )
        self.write_3d_text(Scr.THE_ATTRACTION_FORCE)
        self.write_3d_text(Scr.THIS_IS_DONE_IN_ORD)
        self.remove_3d_text(Scr.THE_ATTRACTION_FORCE, Scr.THIS_IS_DONE_IN_ORD)
        self.write_3d_text(Scr.AFTER_CONVERGENCE)
        self.write_3d_text(Scr.THE_TOP_N_WORDS_WIT)
        self.write_3d_text(Scr.THE_BEST_HINT_FROM)
        self.remove_3d_text(
            Scr.AFTER_CONVERGENCE,
            Scr.THE_TOP_N_WORDS_WIT,
            Scr.THE_BEST_HINT_FROM,
        )
        self.write_3d_text(Scr.HERE_IS_A_GRAPH_OF)
        self.play(FadeOut(text_box), Uncreate(axes), Uncreate(sphere))
        self.plot_guesser_view_chart(r"visualizer\graphs_data\planets.csv", "planets (2 cards)")
        self.plot_guesser_view_chart(r"visualizer\graphs_data\international_good.csv", "international (two cards)")
        self.write_3d_text(Scr.AS_CAN_BE_SEEN2)
        self.write_3d_text(Scr.WITH_SUCH_A_HINT)
        self.remove_3d_text(
            Scr.HERE_IS_A_GRAPH_OF,
            Scr.AS_CAN_BE_SEEN2,
            Scr.WITH_SUCH_A_HINT,
        )
        self.write_3d_text(Scr.HERE_IS_A_GRAPH_OF2)
        self.plot_guesser_view_chart(r"visualizer\graphs_data\dark_bad_choose_it.csv", "dark")
        self.write_3d_text(Scr.AS_CAN_BE_SEEN3)
        self.remove_3d_text(
            Scr.HERE_IS_A_GRAPH_OF2,
            Scr.AS_CAN_BE_SEEN3,
        )
        self.write_3d_text(Scr.SUCH_A_HINT_WILL)

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
            connections = generate_random_connections(vectors_list, average_cluster_size=1.2)
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

    def plot_guesser_view_chart(self, data_path, title):
        df = pd.read_csv(data_path)
        df = df.loc[0:10, :]
        colors = df["colors"].apply(text2color).to_list()
        bar_names = df.iloc[:, 0].to_list()
        horizontal_factor = 1
        CONFIG_dict = {
            "height": 5,
            "width": 9 * horizontal_factor,
            # "n_ticks": 4,
            # "tick_width": 0.2,
            "label_y_axis": "Kaki",
            # "y_axis_label": "Kaki",
            # "y_axis_label_height": 0.25,
            "max_value": 0.5,
            "bar_colors": colors,
            # "bar_fill_opacity": 0.8,
            # "bar_stroke_width": 0,
            "bar_names": bar_names,
            "bar_label_scale_val": 0,
        }

        chart_position_x = 2
        chart_position_y = 0
        labels_size = 0.55
        labels_separation = 0.78 * horizontal_factor
        labels_shift_x = chart_position_x - 3.7
        labels_shift_y = chart_position_y - 3
        bar_labels = VGroup()
        for i in range(len(bar_names)):
            label = Text(bar_names[i])
            self.add_fixed_in_frame_mobjects(label)
            label.scale(labels_size)
            label.move_to(UP * labels_shift_y + (i * labels_separation + labels_shift_x) * RIGHT)
            label.rotate(np.pi * (1.5 / 6))
            bar_labels.add(label)

        chart = BarChart(values=df["distance_to_centroid"].to_list(), **CONFIG_dict)
        chart.shift(chart_position_y * UP + chart_position_x * RIGHT)
        title = Text(f"Hint word: {title}")
        y_label = Text("Cosine distance to hinted word")
        x_label = Text("Board words")
        self.add_fixed_in_frame_mobjects(chart, title, y_label, x_label)
        title.next_to(chart, UP).shift(RIGHT)
        y_label.rotate(angle=TAU / 4, axis=OUT).next_to(chart, LEFT).scale(0.6)
        x_label.next_to(bar_labels, DOWN).scale(0.6)
        # self.add(chart, bar_labels, title, y_label, x_label)
        # self.wait(1)
        self.play(Write(title))
        self.play(DrawBorderThenFill(chart), Write(bar_labels), Write(y_label), Write(x_label), run_time=2)
        self.wait(5)
        self.play(FadeOut(chart), FadeOut(bar_labels), FadeOut(title), FadeOut(y_label), FadeOut(x_label), run_time=2)
        # bar_chart = BarChart(values= df["distance_to_centroid"].to_list(),
        #                      height=7,
        #                      width=9,
        #                      label_y_axis=True,
        #                      bar_colors=colors,
        #                      bar_names= # This is the words
        # self.add_fixed_in_frame_mobjects(bar_chart)
        # self.play(DrawBorderThenFill(bar_chart))

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


# test_scene = KalirmozExplanation()
# starting_point = polar_to_cartesian(1, 0.52 * PI, 1.95 * PI)
# nodes_list = [ForceNode(SKI_VEC, True), ForceNode(WATER_VEC, True), ForceNode(PARK_VEC, False)]
# test_scene.animate_physical_system(
#     starting_point=starting_point, nodes_list=nodes_list, num_of_iterations=1500, arc_radians=0.01
# )
# hint_word = "planets"
# test_scene.plot_guesser_view_chart(f"visualizer\graphs_data\{hint_word}.csv", hint_word)

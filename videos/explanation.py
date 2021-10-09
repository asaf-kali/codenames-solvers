from manim import *
import numpy as np
import random
from typing import Callable, List, Optional
from codenames.solvers.utils.algebra import geodesic, cosine_distance, normalize_vector
from codenames.solvers.sna_solvers.sna_hinter import step_from_forces, ForceNode  # , opponent_force, friendly_force
from scipy.interpolate import interp1d
import pandas as pd


def text2color(text):
    if text == 'blue':
        return BLUE
    if text == 'red':
        return RED
    if text == 'black':
        return DARK_GRAY
    if text == 'gray':
        return GRAY


def random_ints(n: int, k: int) -> List[int]:
    ints_list = [random.randint(0, n) for i in range(k)]
    return ints_list


def geodesic_object(v, u):
    geodesic_function = geodesic(v, u)
    return ParametricFunction(geodesic_function, t_range=[0, 1], color=CONNECTIONS_COLOR)


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
    return 0.6 * (1 / (d + 1) - 0.5)


def generate_progression_dict(titles_texts):
    n = len(titles_texts)
    progression_line = Line(start=ORIGIN + DOWN + PROGRESSION_LINE_LENGTH / 2 * LEFT,
                            end=ORIGIN + DOWN + PROGRESSION_LINE_LENGTH / 2 * RIGHT)
    progression_dots = VGroup()
    for i in range(4):
        dot = AnnotationDot(point=ORIGIN + DOWN + (i - (n-1)/2) * (PROGRESSION_LINE_LENGTH / (n-1)) * RIGHT,
                            fill_color=FUTURE_TOPICS_COLOR,
                            stroke_width=PROGRESSION_DOTS_S_WIDTH)
        progression_dots.add(dot)

    progression_marker = Arrow(start=ORIGIN, end=ORIGIN + UP * 1.5, color=GREEN).next_to(progression_dots[0], DOWN)

    titles = VGroup()
    titles.add(*[Text(titles_texts[i]) for i in range(n)])

    progression_dict = VDict([("line", progression_line),
                              ("dots", progression_dots),
                              ("marker", progression_marker),
                              ("titles", titles)])
    return progression_dict


# def connecting_line(v, u):
#   c = v.T @ u
#   f = lambda t: (t*v+(1-t)*u) / np.sqrt(t**2+(1-t**2) + 2*t*(1-t)*c)
#   return f

BOARD_WORDS = [
    "cloak",
    "kiss",
    "flood", ####
    "mail",
    "skates",
    "paper",
    "frog",
    "house",
    "moon", ####
    "egypt",
    "teacher",
    "storm", ####
    "newton",
    "violet",
    "drill",
    "fever",
    "ninja",
    "jupiter",
    "ski",
    "attic",
    "beach", ####
    "lock",
    "earth",
    "park", ####
    "gymnast"]
FUTURE_TOPICS_COLOR = GRAY
PAST_TOPICS_COLOR = GREEN
CURRENT_TOPICS_COLOR = YELLOW
PROGRESSION_DOTS_S_WIDTH = 2
CARDS_HEIGHT = 1.0
PROGRESSION_LINE_LENGTH = 8
CARDS_WIDTH = 1.6
CARDS_FILL_COLOR = GOLD_E
CARDS_FILL_OPACITY = 0.3
CARDS_HORIZONTAL_SPACING = 0.15
CARDS_VERTICAL_SPACING = 0.3
CARDS_FONT_SIZE = 25
SPHERE_RADIUS = 3
FONT_SIZE_LABELS = 20
FONT_SIZE_TEXT = 25
ARROWS_THICKNESS = 0.001
ARROWS_COLOR = LIGHT_BROWN
DOT_SIZE = 0.2
LABELS_COLOR = PURE_RED
CONNECTIONS_COLOR = ORANGE
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
    ParametricFunction(geodesic(SKI_VEC, WATER_VEC), t_range=[0, 1]).set_color(CONNECTIONS_COLOR),
    ParametricFunction(geodesic(BEACH_VEC, PARK_VEC), t_range=[0, 1]).set_color(CONNECTIONS_COLOR),
    ParametricFunction(geodesic(JUPITER_VEC, NEWTON_VEC), t_range=[0, 1]).set_color(CONNECTIONS_COLOR),
    ParametricFunction(geodesic(TEACHER_VEC, JUPITER_VEC), t_range=[0, 1]).set_color(CONNECTIONS_COLOR),
    ParametricFunction(geodesic(NEWTON_VEC, TEACHER_VEC), t_range=[0, 1]).set_color(CONNECTIONS_COLOR),
]

TITLES_TEXTS = ["Part 1: The Codenames board game",
                "Part 2: The Word2Vec Model",
                "Part 3: The Algorithm",
                "Part 4: Examples"]


class KalirmozExplanation(ThreeDScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.simple_mode = False

    def construct(self):

        progression_dict = generate_progression_dict(TITLES_TEXTS)

        self.scene_intro()

        self.advance_progress_markers(progression_dict, 0, 0)

        self.scene_game_rules()

        # self.advance_progress_markers(progression_dict, 1)

        # self.scene_word2vec_explanation()

        # self.advance_progress_markers(progression_dict, 2)

        # self.scene_sphere()

        # self.add_fixed_orientation_mobjects(progression_dict)
        # self.remove(progression_dict)
        # self.advance_progress_markers(progression_dict, 3)

        # self.scene_guesser_views()

        # self.scene_ending_title()

    def scene_intro(self):
        t1 = Text("Code Names Algorithm", color=BLUE)
        t2 = Text("by the Kali brothers", color=RED).scale(0.8).next_to(t1, DOWN)
        self.write_3d_text(t1)
        self.write_3d_text(t2)
        self.play(FadeOut(t1), FadeOut(t2))

    def scene_game_rules(self):
        blue_hinter = SVGMobject(r"visualizer\Svgs\blue_hinter.svg").to_corner(UR).shift(DOWN).scale(0.8)
        red_hinter = SVGMobject(r"visualizer\Svgs\red_hinter.svg").to_corner(DR).scale(0.8)
        blue_guesser = SVGMobject(r"visualizer\Svgs\blue_guesser.svg").to_corner(UL).align_to(blue_hinter, DOWN).shift(
            LEFT * 0.2)
        red_guesser = SVGMobject(r"visualizer\Svgs\red_guesser.svg").to_corner(DL).shift(LEFT * 0.2)

        board = self.generate_board(BOARD_WORDS)
        self.play(FadeIn(blue_hinter, shift=DOWN),
                  FadeIn(blue_guesser, shift=DOWN),
                  FadeIn(red_hinter, shift=DOWN),
                  FadeIn(red_guesser, shift=DOWN))
        # self.add(board, blue_hinter, blue_guesser, red_hinter, red_guesser)
        self.wait(7)
        self.play(blue_hinter.animate.scale(1.2), red_hinter.animate.scale(1.2), rate_func = there_and_back, run_time = 2)
        self.wait(1)
        self.play(blue_guesser.animate.scale(1.2), red_guesser.animate.scale(1.2), rate_func=there_and_back, run_time=2)
        self.wait(6)
        self.play(DrawBorderThenFill(board))
        self.wait(3)

        self.expose_board_colors(board)




        blue_hinter_bubble, blue_hinter_text = self.animate_hint(blue_hinter, 'planets', 2)

        self.wait(10)

        blue_guesser_bubble = self.animate_guess(board=board,
                                                 card_color=BLUE,
                                                 word='earth',
                                                 guesser_obj=blue_guesser,
                                                 guesser_bubble=None,
                                                 finish_turn=False,
                                                 guess_reveal_wait_time=5)

        self.animate_guess(board=board,
                           card_color=BLUE,
                           word='jupiter',
                           guesser_bubble=blue_guesser_bubble,
                           finish_turn=True)


        self.play(FadeOut(blue_hinter_bubble), FadeOut(blue_hinter_text))

        red_hinter_bubble, red_hinter_text = self.animate_hint(red_hinter, 'water', 4)

        red_guesser_bubble = self.animate_guess(board=board,
                                                card_color=RED,
                                                word='flood',
                                                guesser_obj=red_guesser,
                                                guesser_bubble=None,
                                                finish_turn=False)

        self.animate_guess(board=board,
                           card_color=RED,
                           word='storm',
                           guesser_bubble=red_guesser_bubble,
                           finish_turn=False,
                           quick_mode=True)

        self.animate_guess(board=board,
                           card_color=LIGHT_GRAY,
                           word='moon',
                           guesser_bubble=red_guesser_bubble,
                           finish_turn=True,
                           quick_mode=True)

        self.play(FadeOut(red_hinter_bubble), FadeOut(red_hinter_text))

        blue_hinter_bubble, blue_hinter_text = self.animate_hint(blue_hinter, 'costume', 2)

        blue_guesser_bubble = self.animate_guess(board=board,
                                                 card_color=BLUE,
                                                 word='cloak',
                                                 guesser_obj=blue_guesser,
                                                 guesser_bubble=None,
                                                 finish_turn=False,
                                                 quick_mode=True)

        self.animate_guess(board=board,
                           card_color=DARK_GRAY,
                           word='ninja',
                           guesser_bubble=blue_guesser_bubble,
                           finish_turn=True,
                           quick_mode=True)

        winning_title = Text('Red team Wins!', color=RED).scale(0.8).to_edge(UP).shift(UP * 0.15)
        self.play(Write(winning_title))
        self.wait(2)

        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )

    def animate_hint(self, hinter_obj, hint_word, hint_number):
        hinter_bubble = SVGMobject(r"visualizer\Svgs\centered.svg").scale(0.5).next_to(hinter_obj, UP)
        hinter_text = Text(f'"{hint_word}", {hint_number}', font_size=18).move_to(hinter_bubble).shift(UP * 0.2)
        self.play(FadeIn(hinter_bubble, shift=DOWN))
        self.play(Write(hinter_text))
        self.wait(3)
        return hinter_bubble, hinter_text

    def animate_guess(self, board, card_color, word, guesser_obj=None, guesser_bubble=None, finish_turn=False,
                      guess_reveal_wait_time=1, quick_mode=False):
        if quick_mode is True:
            quick_factor = 0.5
        else:
            quick_factor = 1
        if guesser_bubble is None:
            guesser_bubble = SVGMobject(r"visualizer\Svgs\centered.svg").scale(0.6).next_to(guesser_obj, UP).shift(
                DOWN * 0.3)
            self.play(FadeIn(guesser_bubble, shift=DOWN), run_time=quick_factor)
        guesser_text = Text(f'I Guess: "{word}"', font_size=17).move_to(guesser_bubble).shift(UP * 0.2)
        if quick_mode:
            self.play(FadeIn(guesser_text), run_time=quick_factor)

        else:
            self.play(Write(guesser_text))
        self.wait(guess_reveal_wait_time * quick_factor)
        self.change_card_color(board, word, card_color, annotate=quick_mode)
        self.wait(quick_factor)
        if finish_turn:
            self.play(FadeOut(guesser_text), FadeOut(guesser_bubble), run_time=quick_factor)
        else:
            self.play(FadeOut(guesser_text), run_time=quick_factor)
        return guesser_bubble

    def advance_progress_markers(self, progression_dict, i, previous_i=None, extra_waiting_time=1):
        if previous_i is None:
            previous_i = i-1

        progression_dict["dots"][previous_i].set_fill(CURRENT_TOPICS_COLOR)
        for j in range(previous_i):
            progression_dict["dots"][j].set_fill(PAST_TOPICS_COLOR)
        for j in range(previous_i + 1, len(progression_dict["dots"])):
            progression_dict["dots"][j].set_fill(FUTURE_TOPICS_COLOR)
        progression_dict["marker"].next_to(progression_dict["dots"][previous_i], DOWN)

        self.play(FadeIn(progression_dict["marker"]),
                  FadeIn(progression_dict["line"]),
                  FadeIn(progression_dict["dots"]),
                  FadeIn(progression_dict["titles"][previous_i]))

        self.wait(1)

        if previous_i != i:
            self.play(progression_dict["marker"].animate.next_to(progression_dict["dots"][i], DOWN),
                      *[progression_dict["dots"][j].animate.set_fill(PAST_TOPICS_COLOR) for j in range(i)],
                      *[progression_dict["dots"][j].animate.set_fill(FUTURE_TOPICS_COLOR) for j in range(i+1,len(progression_dict["dots"]))],
                      progression_dict["dots"][i].animate.set_fill(CURRENT_TOPICS_COLOR),
                      FadeOut(progression_dict["titles"][previous_i], shift=DOWN),
                      FadeIn(progression_dict["titles"][i], shift=DOWN))
            self.wait(1)
        self.wait(extra_waiting_time)

        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )

    def change_card_color(self, board, word, color, annotate=False):
        card_index = BOARD_WORDS.index(word)
        if annotate:
            self.play(board[card_index].animate.scale(1.1), rate_func=there_and_back, run_time=2)
        self.play(board[card_index][0].animate.set_color(color))

    def scene_sphere(self):
        theta = 240 * DEGREES
        phi = 75 * DEGREES
        axes = ThreeDAxes(
            x_range=[-1.5, 1.5, 1], y_range=[-1.5, 1.5, 1], z_range=[-1.5, 1.5, 1], x_length=8, y_length=8, z_length=8
        )
        sphere = Sphere(
            center=(0, 0, 0), radius=SPHERE_RADIUS, resolution=(20, 20), u_range=[0.001, PI - 0.001], v_range=[0, TAU]
        ).set_opacity(0.3)

        self.renderer.camera.light_source.move_to(3 * IN)
        self.set_camera_orientation(phi=phi, theta=theta)  # 75 * DEGREES, theta=30 * DEGREES
        self.begin_ambient_camera_rotation(rate=0.1)
        self.play(Create(axes), Create(sphere))
        self.add(axes, sphere)
        # self.wait(1)

        words_labels_list = [
            self.generate_card(text=labels_list[i],
                               height=0.4,
                               width=1.4,
                               fill_color=CARDS_FILL_COLOR,
                               fill_opacity=0.4,
                               font_size=CARDS_FONT_SIZE,
                               stroke_color=GOLD_E).move_to(vectors_list[i]*1.1)
            # Text(labels_list[i], font_size=FONT_SIZE_LABELS, color=LABELS_COLOR).move_to(vectors_list[i])
            for i in range(words_list_len)
        ]
        arrows_list = [Line(start=[0, 0, 0], end=vector, color=ARROWS_COLOR) for vector in
                       vectors_list]  # stroke_width=ARROWS_THICKNESS
        self.play(*[Create(arrows_list[i]) for i in range(words_list_len)])
        for i in range(words_list_len):
            self.add_fixed_orientation_mobjects(words_labels_list[i])
            # words_labels_list[i].add_updater(lambda x, i=i: x.move_to(self.coords_to_point(vectors_list[i])))
        self.play(*[FadeIn(words_labels_list[i]) for i in range(words_list_len)])
        self.wait(3)
        # self.add(*arrows_list, *words_labels_list)

        self.play(*[Create(connection, run_time=3) for connection in sna_connections_list])
        self.wait(5)
        self.play(*[Uncreate(connection, run_time=1) for connection in sna_connections_list])
        self.wait(4)

        self.animate_random_connections(vectors_list=vectors_list, number_of_examples=15, example_length=0.3)
        self.wait(4)

        starting_point = polar_to_cartesian(SPHERE_RADIUS, 0.52 * PI, 1.95 * PI)
        nodes_list = [ForceNode(SKI_VEC, True), ForceNode(WATER_VEC, True), ForceNode(PARK_VEC, False),
                      ForceNode(TEACHER_VEC, False)]
        self.animate_physical_system(
            starting_point=starting_point, nodes_list=nodes_list, num_of_iterations=1000, arc_radians=0.01, run_time=5
        )
        self.wait(2)

        self.play(Uncreate(axes),
                  Uncreate(sphere),
                  *[Uncreate(arrows_list[i]) for i in range(words_list_len)],
                  *[FadeOut(words_labels_list[i]) for i in range(words_list_len)])  # , FadeOut(text_box)
        self.stop_ambient_camera_rotation()

    def scene_guesser_views(self):
        self.plot_guesser_view_chart(r"visualizer\graphs_data\planets.csv", "planets (2 cards)")
        self.plot_guesser_view_chart(r"visualizer\graphs_data\international_good.csv", 'international (two cards)')
        self.plot_guesser_view_chart(r"visualizer\graphs_data\dark_bad_choose_it.csv", 'dark')
        # self.plot_guesser_view_chart(r"visualizer\graphs_data\apollo_bad.csv", 'apollo')
        # self.plot_guesser_view_chart(r"visualizer\graphs_data\rhino_bad.csv", 'rhino')
        # self.plot_guesser_view_chart(r"visualizer\graphs_data\rhino_bad.csv", 'rhino')
        # self.plot_guesser_view_chart(r"visualizer\graphs_data\advice.csv", "advice")
        # self.plot_guesser_view_chart(r"visualizer\graphs_data\beneath_good.csv", 'beneath')
        # self.plot_guesser_view_chart(r"visualizer\graphs_data\ceilings_good.csv", 'ceillings')
        # self.plot_guesser_view_chart(r"visualizer\graphs_data\father.csv", 'father')
        # self.plot_guesser_view_chart(r"visualizer\graphs_data\gear_good.csv", 'gear')
        # self.plot_guesser_view_chart(r"visualizer\graphs_data\redevelopment_fine.csv", 'redevelopment')

        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )

    def scene_ending_title(self):
        ending_title = Text("Thanks for watching!")
        self.write_3d_text(ending_title)
        self.wait(2)
        self.play(FadeOut(ending_title))
        self.wait(1)

    @staticmethod
    def specific_color_map(i):
        if i in [0, 4, 9, 10, 14, 18, 22, 24]:
            return RED
        elif i in [2, 6, 8, 11, 16, 17, 20, 23, 25]:
            return BLUE
        elif i == 19:
            return DARK_GRAY
        else:
            return LIGHT_GRAY

    def expose_board_colors(self, board):
        self.play(*[board[i][0].animate.set_color(self.specific_color_map(i)) for i in range(25)])
        self.wait(11)
        self.play(*[board[i][0].animate.set_color(CARDS_FILL_COLOR) for i in range(25)])
        self.wait(14)

    def generate_board(self, words_list):
        board = VGroup()
        for r in range(5):
            for c in range(5):
                current_card = self.generate_card(words_list[5 * r + c])
                current_card.move_to(ORIGIN +
                                     (r - 2.5) * CARDS_HEIGHT * (1 + CARDS_VERTICAL_SPACING) * UP +
                                     (c - 2.5) * CARDS_WIDTH * (1 + CARDS_HORIZONTAL_SPACING) * RIGHT)
                board.add(current_card)
        board.move_to(ORIGIN)
        return board

    @staticmethod
    def generate_card(text, height=CARDS_HEIGHT, width=CARDS_WIDTH, fill_color=CARDS_FILL_COLOR,
                      fill_opacity=CARDS_FILL_OPACITY, font_size=CARDS_FONT_SIZE, stroke_color=None):
        card = VGroup()
        rectangle = Rectangle(height=height,
                              width=width,
                              fill_color=fill_color,
                              fill_opacity=fill_opacity,
                              stroke_color=stroke_color)
        text = Text(text, font_size=font_size).move_to(rectangle)
        card.add(rectangle, text)
        return card

    def animate_game_explanation(self):
        self.play(
            FadeIn(SVGMobject(r"visualizer\Svgs\hinter.svg"))
        )

    def scene_word2vec_explanation(self):
        spacing_constant = 3
        lion_color = BLUE
        deer_color = RED
        national_color = GREEN
        arrows_scale = 1
        words_horizontal_shift = 2
        word2vec_title = Text('Part Two: The Word2Vec Model').scale(1.2).to_edge(UP)
        self.play(Write(word2vec_title))

        lion_text = Text('lion:', color=lion_color).shift(words_horizontal_shift * LEFT)
        deer_text = Text('deer:', color=deer_color).shift(words_horizontal_shift * LEFT)
        nationalism_text = Text('nationalism:', color=national_color).shift(words_horizontal_shift * LEFT)
        lion_theta = 0.45 * PI
        deer_theta = 0.54 * PI
        nationalism_theta = 3 / 2 * PI
        lion_vec = arrows_scale * (np.cos(lion_theta) * RIGHT + np.sin(lion_theta) * UP)
        deer_vec = arrows_scale * (np.cos(deer_theta) * RIGHT + np.sin(deer_theta) * UP)
        nationalism_vec = arrows_scale * (np.cos(nationalism_theta) * RIGHT + np.sin(nationalism_theta) * UP)
        lion_arrow = Arrow(start=ORIGIN, end=lion_vec).next_to(lion_text, spacing_constant * RIGHT).set_color(
            lion_color)
        lion_arrow.add_updater(lambda x: x.next_to(lion_text, RIGHT))
        deer_arrow = Arrow(start=ORIGIN, end=deer_vec).next_to(deer_text, spacing_constant * RIGHT).set_color(
            deer_color)
        deer_arrow.add_updater(lambda x: x.next_to(deer_text, RIGHT))
        nationalism_arrow = Arrow(start=ORIGIN,
                                  end=nationalism_vec).next_to(nationalism_text, spacing_constant * RIGHT).set_color(
            national_color)
        nationalism_arrow.add_updater(lambda x: x.next_to(nationalism_text, RIGHT))

        self.play(FadeIn(lion_text))
        self.play(Create(lion_arrow))
        self.wait(2)
        self.play(lion_text.animate.shift(1.5 * UP))
        deer_text.next_to(lion_text, spacing_constant * DOWN)
        nationalism_text.next_to(deer_text, spacing_constant * DOWN)
        self.play(FadeIn(deer_text, shift=UP), FadeIn(deer_arrow, shift=UP))
        self.play(FadeIn(nationalism_text, shift=UP), FadeIn(nationalism_arrow, shift=UP))
        lion_arrow.clear_updaters()
        deer_arrow.clear_updaters()
        nationalism_arrow.clear_updaters()
        arrows_anchor = 4 * RIGHT
        self.play(lion_arrow.animate.put_start_and_end_on(start=arrows_anchor, end=arrows_anchor + lion_vec),
                  deer_arrow.animate.put_start_and_end_on(start=arrows_anchor, end=arrows_anchor + deer_vec),
                  nationalism_arrow.animate.put_start_and_end_on(start=arrows_anchor,
                                                                 end=arrows_anchor + nationalism_vec))
        self.wait(2)
        small_angle = Angle(lion_arrow, deer_arrow, radius=0.6 * arrows_scale)
        big_angle = Angle(nationalism_arrow, lion_arrow, radius=0.3 * arrows_scale)
        self.play(Create(small_angle))
        self.play(Create(big_angle))
        self.play(FadeOut(word2vec_title),
                  FadeOut(deer_text),
                  FadeOut(nationalism_text),
                  FadeOut(lion_arrow),
                  FadeOut(deer_arrow),
                  FadeOut(nationalism_arrow),
                  FadeOut(big_angle),
                  FadeOut(small_angle),
                  FadeOut(lion_text))

    def animate_physical_system(
            self, starting_point: np.array, nodes_list, num_of_iterations=10, arc_radians=0.01, run_time=7
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
        centroid_dot = Dot(point=starting_point)
        self.add_fixed_orientation_mobjects(centroid_dot)
        centroid.add_updater(lambda x: x.become(Line(start=[0, 0, 0], end=trajectory_interp(t.get_value()))))
        centroid_dot.add_updater(lambda x: x.become(Dot(point=trajectory_interp(t.get_value()))))
        # forces = []
        # for i, node in enumerate(nodes_list):
        #     force = geodesic_object(node.force_origin, normalize_vector(centroid.get_end()))
        #     if node.force_sign:
        #         force.set_color(BLUE)
        #     else:
        #         force.set_color(RED)
        #     force.add_updater(lambda x, i=i: x.become(geodesic_object(node.force_origin, normalize_vector(trajectory_interp(t.get_value())))))
        #     forces.append(force)
        # This is an explicit version of the forl
        park_force = geodesic_object(PARK_VEC, normalize_vector(centroid.get_end())).set_color(RED)
        park_force.add_updater(
            lambda x: x.become(geodesic_object(PARK_VEC, normalize_vector(trajectory_interp(t.get_value())))).set_color(
                RED)
        )
        teacher_force = geodesic_object(TEACHER_VEC, normalize_vector(centroid.get_end())).set_color(RED)
        teacher_force.add_updater(
            lambda x: x.become(
                geodesic_object(TEACHER_VEC, normalize_vector(trajectory_interp(t.get_value())))).set_color(RED)
        )
        ski_force = geodesic_object(SKI_VEC, normalize_vector(centroid.get_end())).set_color(BLUE)
        ski_force.add_updater(
            lambda x: x.become(geodesic_object(SKI_VEC, normalize_vector(trajectory_interp(t.get_value())))).set_color(
                BLUE)
        )
        water_force = geodesic_object(WATER_VEC, normalize_vector(centroid.get_end())).set_color(BLUE)
        water_force.add_updater(
            lambda x: x.become(
                geodesic_object(WATER_VEC, normalize_vector(trajectory_interp(t.get_value())))).set_color(BLUE)
        )
        self.play(Create(centroid), Create(centroid_dot), Create(park_force), Create(ski_force), Create(water_force),
                  Create(teacher_force))
        self.play(t.animate.set_value(1), run_time=run_time, rate_func=linear)
        self.play(FadeOut(centroid, centroid_dot, park_force, ski_force, water_force, teacher_force))

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
            "bar_label_scale_val": 0

        }

        chart_position_x = 0
        chart_position_y = 0.2
        labels_size = 0.55
        labels_separation = 0.78 * horizontal_factor
        labels_shift_x = chart_position_x - 3.7
        labels_shift_y = chart_position_y - 3
        bar_labels = VGroup()
        for i in range(len(bar_names)):
            label = Text(bar_names[i])
            label.scale(labels_size)
            label.move_to(UP * labels_shift_y + (i * labels_separation + labels_shift_x) * RIGHT)
            label.rotate(np.pi * (1.5 / 6))
            bar_labels.add(label)

        chart = BarChart(values=df["distance_to_centroid"].to_list(), **CONFIG_dict)
        chart.shift(chart_position_y * UP + chart_position_x * RIGHT)
        title = Text(f"Hint word: {title}")
        y_label = Text("Cosine distance to hinted word")
        x_label = Text("Board words")
        title.next_to(chart, UP).shift(RIGHT-chart_position_y*UP)
        y_label.rotate(angle=TAU / 4, axis=OUT).next_to(chart, LEFT).scale(0.6)
        x_label.next_to(bar_labels, DOWN).scale(0.6).shift(0.2 * UP)

        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        self.add_fixed_in_frame_mobjects(chart, y_label, x_label, bar_labels)
        self.play(DrawBorderThenFill(chart),
                  FadeIn(bar_labels, shift=UP),
                  FadeIn(y_label, shift=UP),
                  FadeIn(x_label, shift=UP),
                  run_time=1)
        self.wait(5)
        self.play(FadeOut(chart), FadeOut(bar_labels), FadeOut(title), FadeOut(y_label), FadeOut(x_label),
                  run_time=1)

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
# test_scene.construct()
# starting_point = polar_to_cartesian(1, 0.52 * PI, 1.95 * PI)
# nodes_list = [ForceNode(SKI_VEC, True), ForceNode(WATER_VEC, True), ForceNode(PARK_VEC, False)]
# test_scene.animate_physical_system(
#     starting_point=starting_point, nodes_list=nodes_list, num_of_iterations=5, arc_radians=0.01
# )
# hint_word = "planets"
# test_scene.plot_guesser_view_chart(f"visualizer\graphs_data\{hint_word}.csv", hint_word)

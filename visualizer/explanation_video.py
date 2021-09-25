from manim import *
import numpy as np


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


def polar_to_cartesian(r, phi, theta):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.array([x, y, z])


# def connecting_line(v, u):
#   c = v.T @ u
#   f = lambda t: (t*v+(1-t)*u) / np.sqrt(t**2+(1-t**2) + 2*t*(1-t)*c)
#   return f

def single_gram_schmidt(v: np.ndarray, u: np.ndarray):
    v = v / np.linalg.norm(v)
    u = u / np.linalg.norm(u)

    projection_norm = u.T @ v

    o = u - projection_norm * v

    normed_o = o / np.linalg.norm(o)
    return v, normed_o


def connecting_line(v, u):
    v, normed_o = single_gram_schmidt(v, u)
    theta = np.arccos(v.T @ u)
    f = lambda t: np.cos(t * theta) * v + np.sin(t * theta) * normed_o
    return f


SPHERE_RADIUS = 1
FONT_SIZE_LABELS = 12
FONT_SIZE_TEXT = 12
ARROWS_WIDTH = 0.001
DOT_SIZE = 0.2
KING_VEC = polar_to_cartesian(SPHERE_RADIUS, 0.5 * PI, 0)
QUEEN_VEC = polar_to_cartesian(SPHERE_RADIUS, 0.5 * PI, 0.2 * PI)
BEACH_VEC = polar_to_cartesian(SPHERE_RADIUS, 0.70 * PI, 1.17 * PI)
PARK_VEC = polar_to_cartesian(SPHERE_RADIUS, 0.75 * PI, 1.25 * PI)
JUPITER_VEC = polar_to_cartesian(SPHERE_RADIUS, 0.2 * PI, -0.3 * PI)
NEWTON_VEC = polar_to_cartesian(SPHERE_RADIUS, 0.27 * PI, -0.11 * PI)
TEACHER_VEC = polar_to_cartesian(SPHERE_RADIUS, 0.3 * PI, -0.2 * PI)

vectors_list = [KING_VEC, QUEEN_VEC, BEACH_VEC, PARK_VEC, JUPITER_VEC, NEWTON_VEC, TEACHER_VEC]
labels_list = ['king', 'queen', 'beach', 'park', 'jupiter', 'newton', 'teacher']
list_len = len(vectors_list)
connections_list = [ParametricFunction(connecting_line(KING_VEC, QUEEN_VEC), t_range=[0, 1]),
                    ParametricFunction(connecting_line(BEACH_VEC, PARK_VEC), t_range=[0, 1]),
                    ParametricFunction(connecting_line(JUPITER_VEC, NEWTON_VEC), t_range=[0, 1]),
                    ParametricFunction(connecting_line(TEACHER_VEC, JUPITER_VEC), t_range=[0, 1]),
                    ParametricFunction(connecting_line(NEWTON_VEC, TEACHER_VEC), t_range=[0, 1])

                    ]


class KalirmozExplanation(ThreeDScene):
    def construct(self):
        theta = 1.3 * PI / 2
        phi = 0.8 * PI
        axes = ThreeDAxes(x_range=[-2, 2, 1], x_length=4, y_length=4)
        # labels = axes.get_axis_labels(
        #     x_label=Tex("x"), y_label=Tex("y")
        # )
        sphere = Sphere(
            center=(0, 0, 0),
            radius=SPHERE_RADIUS,
            resolution=(20, 20),
            u_range=[0.001, PI - 0.001],
            v_range=[0, TAU]
        ).set_opacity(0.3)
        t1 = Text('In each turn, the first task of the hinter is to\nfind a proper subset of words (usualy '
                  'two to four words), on which to hint', font_size=FONT_SIZE_TEXT).to_corner(UL)
        t2 = Text('Two methods of clustering where implemented.', font_size=FONT_SIZE_TEXT).to_corner(UL)
        t3 = Text('In the first clustering method, the words are considered as\n nodes in a graph, with edges '
                  'wheigt corrolated\n to their cosine similarity', font_size=FONT_SIZE_TEXT).to_corner(UL)
        t4 = Text('This graph is devided into communities using\nthe louvain algorithm, and each community is taken'
                  '\nas an optional cluster of words to hint about.', font_size=FONT_SIZE_TEXT).to_corner(UL)
        arrows_list = [Arrow3D(start=[0, 0, 0], end=vector) for vector in vectors_list]
        camera_vectos = [self.coords_to_point(vector * 1.2) for vector in vectors_list]
        texts_list = [Text(labels_list[i], font_size=FONT_SIZE_LABELS, ).move_to(camera_vectos[i]) for i in
                      range(list_len)]
        # dots_list = [Dot(point=vector, radius=DOT_SIZE) for vector in vectors_list]

        self.renderer.camera.light_source.move_to(3 * IN)  # changes the source of the light
        self.set_camera_orientation(phi=phi, theta=theta)  # 75 * DEGREES, theta=30 * DEGREES

        self.add(axes, sphere)
        # self.play(*[Create(arrows_list[i]) for i in range(list_len)])
        self.add_fixed_in_frame_mobjects(t1, t2, t3, t4)
        updaters = [lambda mob: self.update_position_to_camera(mob, vector) for vector in vectors_list]
        for i in range(list_len):
            self.add_fixed_in_frame_mobjects(texts_list[i])
        texts_list[0].add_updater(lambda x: x.move_to(self.coords_to_point(vectors_list[0])))
        texts_list[1].add_updater(lambda x: x.move_to(self.coords_to_point(vectors_list[1])))
        texts_list[2].add_updater(lambda x: x.move_to(self.coords_to_point(vectors_list[2])))
        texts_list[3].add_updater(lambda x: x.move_to(self.coords_to_point(vectors_list[3])))
        texts_list[4].add_updater(lambda x: x.move_to(self.coords_to_point(vectors_list[4])))
        texts_list[5].add_updater(lambda x: x.move_to(self.coords_to_point(vectors_list[5])))
        texts_list[6].add_updater(lambda x: x.move_to(self.coords_to_point(vectors_list[6])))
        self.add(*arrows_list[0:list_len])
        self.play(Write(t1))
        self.play(FadeOut(t1))
        self.play(Write(t2))
        self.play(FadeOut(t2))
        self.play(Write(t3))
        self.play(FadeOut(t3))
        self.play(Write(t4))
        self.play(FadeOut(t4))
        # self.begin_ambient_camera_rotation(rate=1)
        # self.wait(1)

        # self.play(*[Create(connection, run_time=5) for connection in connections_list])
        # self.wait(2)

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

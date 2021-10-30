from manim import DOWN, Scene, Text, Write
from manimlib import BLUE, RED


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

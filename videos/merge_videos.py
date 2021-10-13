from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip
import subprocess
# %%
manim_video = VideoFileClip("media/videos/explanation/1080p60/KalirmozExplanation.mp4")
graphviz_video = VideoFileClip("videos/lynx_example.mp4")
# %%
clip_1 = manim_video.subclip(1,4)
clip_2 = manim_video.subclip(1,2)
clip_3 = manim_video.subclip(4,6)


# %%

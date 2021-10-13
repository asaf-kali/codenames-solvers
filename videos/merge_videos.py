from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip
import subprocess
# %%
manim_video = VideoFileClip("media/videos/explanation/1080p60/KalirmozExplanation.mp4")
graphviz_video = VideoFileClip("videos/lynx_example.mp4")
# %%
clip_1 = manim_video.subclip(0,1)
clip_2 = manim_video.subclip(1,2)
clip_3 = manim_video.subclip(1,2)

composited_clip = CompositeVideoClip([clip_1, clip_2, clip_3])

composited_clip.write_videofile(f"composite_test.mp4", fps=clip_1.fps, remove_temp=True, codec="libx264") # , temp_audiofile="temp-audio.m4a" , audio_codec="aac"


# %%

# Import everything needed to edit video clips
from moviepy.editor import *

# loading video dsa gfg intro video
clip = VideoFileClip("polyp.mp4")

# getting subclip as video is large
clip1 = clip.subclip(10, 20)

# getting subclip as video is large
clip2 = clip.subclip(80, 100)


final = concatenate_videoclips([clip1, clip2])
final.write_videofile("output1.mp4")





from moviepy.editor import *

# loading video dsa gfg intro video
clip = VideoFileClip("output.mp4")

# getting subclip as video is large
clip1 = clip.subclip(0, 5)
duration = clip1.duration
print(duration)

# loading video gfg
clipx = VideoFileClip("polyp.mp4")

# getting subclip
clip2 = clipx.subclip(0, 5)

# clip list
clips = [clip1, clip2]

# concatenating both the clips
final = concatenate_videoclips(clips)
final.write_videofile("output2.mp4")
# showing final clip
# final.ipython_display(width = 480)


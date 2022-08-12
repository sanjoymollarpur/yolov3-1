import moviepy.editor as mp

audio = mp.AudioFileClip("audio.mpga")
video1 = mp.VideoFileClip("polyp.mp4")
final = video1.set_audio(audio)

final.write_videofile("output.mp4")
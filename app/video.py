import moviepy.editor as mp

from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
import os
import time

# For windows, the below if condition is must.
def generate_key_frames(video_file_path):
    start_time = time.time()
    # initialize video module
    vd = Video()

    # number of images to be returned
    no_of_frames_to_returned = 6

    # initialize diskwriter to save data at desired location
    disk_writer = KeyFrameDiskWriter(location="tmp/key_frames", file_ext=".jpg")

    print(f"Input video file path = {video_file_path}")

    # extract keyframes and process data with diskwriter
    key_frames = vd.extract_video_keyframes(
        no_of_frames=no_of_frames_to_returned,
        file_path=video_file_path,
        writer=disk_writer
    )

    print("--- %s seconds ---" % (time.time() - start_time))

    return key_frames

def get_audio(video_file_path):
    
    audio_path = "tmp/audio_track.mp3"
    my_clip = mp.VideoFileClip(video_file_path)
    my_clip.audio.write_audiofile(audio_path)

    return audio_path

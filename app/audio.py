
# import speech_recognition as sr

import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
import whisper
import numpy as np

from multiprocessing.pool import ThreadPool as Pool

import threading
import traceback

pool_size = 4  # your "parallelness"
lock = threading.Lock()

# define worker function before a Pool is instantiated
# def worker(chunk, i, arr):
#     model = load_model()
#     try:
#         chunk_silent = AudioSegment.silent(duration = 10)

#         # add 0.5 sec silence to beginning and
#         # end of audio chunk. This is done so that
#         # it doesn't seem abruptly sliced.
#         audio_chunk = chunk_silent + chunk + chunk_silent

#         # export audio chunk and save it in
#         # the current directory.
#         print("saving chunk{0}.mp3".format(i))
#         # specify the bitrate to be 192 k
#         audio_chunk.export("./tmp/chunk{0}.mp3".format(i), bitrate ='192k', format ="mp3")
#         filename = 'chunk'+str(i)+'.mp3'

#         result = model.transcribe(filename)

#         print("chunk{0}.mp3 transcribed".format(i))
#         with lock:
#             arr[i] = result["text"]
#         print("chunk{0}.mp3 done".format(i))
#     except Exception as e:
#         traceback.print_exc()
#     del model

def load_model():
    return whisper.load_model("base")

def silence_based_conversion(audio_file, model):

    # open the audio file stored in
    # the local system as a wav file.
    # song, sr = librosa.load(path)
    normalized = False
    # sf.write('stereo_file.mp3', audio_file[1], audio_file[0])
    channels = 2 if (audio_file[1].ndim == 2 and audio_file[1].shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(audio_file[1] * 2 ** 15)
    else:
        y = np.int16(audio_file[1])

    song = AudioSegment(y.tobytes(), frame_rate=audio_file[0], sample_width=2, channels=channels)
    
    # song = AudioSegment.from_file('stereo_file.mp3', format="mp3")

    # open a file where we will concatenate
    # and store the recognized text
    # fh = open("recognized.txt", "w+")

    # split track where silence is 0.5 seconds
    # or more and get chunks
    chunks = split_on_silence(song,
                              # must be silent for at least 1 second
                              # or 1000 ms. adjust this value based on user
                              # requirement. if the speaker stays silent for
                              # longer, increase this value. else, decrease it.
                              min_silence_len = 750,

                              # consider it silent if quieter than -16 dBFS
                              # adjust this per requirement
                              silence_thresh = -40
                              )

    # create a directory to store the audio chunks.
    try:
        os.mkdir('tmp/audio_chunks')
    except(FileExistsError):
        pass

    # move into the directory to
    # store the audio files.
    os.chdir('tmp')

    print(len(chunks))
    # i = 0
    res = ''
    
    pool = Pool(pool_size)

    arr = np.zeros(len(chunks))
    model = load_model()
    # process each chunk
    for i, chunk in enumerate(chunks):

        # pool.apply_async(worker, (chunk, i, arr))

        # Create 0.5 seconds silence chunk
        chunk_silent = AudioSegment.silent(duration = 10)

        # add 0.5 sec silence to beginning and
        # end of audio chunk. This is done so that
        # it doesn't seem abruptly sliced.
        audio_chunk = chunk_silent + chunk + chunk_silent

        # export audio chunk and save it in
        # the current directory.
        print("saving chunk{0}.mp3".format(i))
        # specify the bitrate to be 192 k
        audio_chunk.export("./audio_chunks/chunk{0}.mp3".format(i), bitrate ='192k', format ="mp3")

        # the name of the newly created chunk
        filename = f'./audio_chunks/chunk{str(i)}.mp3'

        result = model.transcribe(filename)
        res = res + result["text"]

    pool.close()
    pool.join()

    os.chdir('..')
    # os.rmtree('tmp_audio_chunks')

    return res
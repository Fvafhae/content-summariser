
# import speech_recognition as sr

import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
import whisper

import librosa
import soundfile as sf

def load_model():
    return whisper.load_model("base")

def silence_based_conversion(audio_file, model):

    # open the audio file stored in
    # the local system as a wav file.
    # song, sr = librosa.load(path)
    sf.write('stereo_file.wav', audio_file[1], audio_file[0])
    song = AudioSegment.from_file('stereo_file.wav', format="wav")

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
        os.mkdir('tmp_audio_chunks')
    except(FileExistsError):
        pass

    # move into the directory to
    # store the audio files.
    os.chdir('tmp_audio_chunks')

    print(len(chunks))
    # i = 0
    res = ''

    # process each chunk
    for i, chunk in enumerate(chunks):

        # Create 0.5 seconds silence chunk
        chunk_silent = AudioSegment.silent(duration = 10)

        # add 0.5 sec silence to beginning and
        # end of audio chunk. This is done so that
        # it doesn't seem abruptly sliced.
        audio_chunk = chunk_silent + chunk + chunk_silent

        # export audio chunk and save it in
        # the current directory.
        print("saving chunk{0}.wav".format(i))
        # specify the bitrate to be 192 k
        audio_chunk.export("./chunk{0}.wav".format(i), bitrate ='192k', format ="wav")

        # the name of the newly created chunk
        filename = 'chunk'+str(i)+'.wav'

        result = model.transcribe(filename)
        res = res + result["text"]

    os.chdir('..')
    # os.rmtree('tmp_audio_chunks')

    return res
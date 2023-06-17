# https://pypi.org/project/audioread/

import audioread

def load_file(filename):
    with audioread.audio_open(filename) as f:
        print(f.channels, f.samplerate, f.duration)
        return f

f = load_file("../test.mp3")
import random
from audio_augmentation import AudioAugmentation
from functions import equalizer

class Equalizer(AudioAugmentation):
    def __init__(self, gain_min, gain_max):
        self.gain_min = gain_min
        self.gain_max = gain_max

    def apply(self, input_file, output_file, gains):
        # Apply Equalizer augmentation here
        equalizer (input_file, output_file , gains[0], gains[1], gains[2], gains[3], gains[4], gains[5], gains[6])

    def randomize(self):
        gains = [random.uniform(self.gain_min, self.gain_max) for _ in range(7)]
        return gains,

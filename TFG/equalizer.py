import random
from audio_augmentation import AudioAugmentation
from functions import equalizer

class Equalizer(AudioAugmentation):
    def __init__(self, gain_min, gain_max):
        super().__init__(gain_min, gain_max)

    def apply(self, input_file, output_file):
        # Apply Equalizer augmentation here
        equalizer (input_file, output_file , gain1=0, gain2=0, gain3=0, gain4=0, gain5=0, gain6=0, gain7=0)

    def randomize(self):
        gains = [random.uniform(self.gain_min, self.gain_max) for _ in range(7)]
        return gains

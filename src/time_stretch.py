from audio_augmentation import AudioAugmentation
from functions import time_stretch
import random
class Time_stretch(AudioAugmentation):
    def __init__(self, factor):
        self.factor = [
            (1.2, 'easy'),
            (1.15, 'easy'),
            (1.1, 'easy'),
            (0.9, 'easy'),
            (0.95, 'easy'),
            (0.8, 'easy')
        ]
        self.factor = factor

    def apply(self, input_file, output_file, factor):
        time_stretch(input_file, output_file, factor)

    def randomize(self):
        speed_factor = random.choice(self.factor)
        return speed_factor
from audio_augmentation import AudioAugmentation
from functions import time_stretch
import random
class Time_stretch(AudioAugmentation):
    def __init__(self, min_factor , max_factor):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def apply(self, input_file, output_file, pitch_factor):
        time_stretch(input_file, output_file, pitch_factor)

    def randomize(self):
        speed_factor = random.uniform(self.min_factor, self.max_factor)
        return speed_factor,
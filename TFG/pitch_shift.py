from audio_augmentation import AudioAugmentation
from functions import pitch_shift
import random
class Pitch_shift(AudioAugmentation):
    def __init__(self, min_shift , max_shift):
        self.min_shift = min_shift
        self.max_shift = max_shift

    def apply(self, input_file, output_file, pitch_factor):
        pitch_shift(input_file, output_file, pitch_factor)

    def randomize(self):
        pitch_factor = random.uniform(self.min_shift, self.max_shift)
        return pitch_factor,
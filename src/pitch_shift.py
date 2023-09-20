from audio_augmentation import AudioAugmentation
from functions import pitch_shift
import random
class Pitch_shift(AudioAugmentation):
    def __init__(self, shift):
        self.shift = [
            (1.2, 'easy'),
            (1.15, 'easy'),
            (1.1, 'easy'),
            (0.9, 'easy'),
            (0.95, 'easy'),
            (0.8, 'easy')
        ]
        self.shift = shift

    def apply(self, input_file, output_file, pitch_shift):
        pitch_shift(input_file, output_file, pitch_shift)

    def randomize(self):
        shift = random.choice(self.factor)
        return shift
import random
from audio_augmentation import AudioAugmentation
from functions import clipping

class Clipping(AudioAugmentation):
    def __init__(self):
        self.clipping_values = [
            (1.0, 'easy'),
            (5.0, 'easy'),
            (10.0, 'medium'),
            (20.0, 'medium'),
            (30.0, 'hard'),
            (40.0, 'hard')
        ]

    def apply(self, input_file, output_file, percentile_threshold):
        clipping(input_file, output_file, percentile_threshold)

    def randomize(self):
        random_pair = random.choice(self.clipping_values)
        return random_pair


    
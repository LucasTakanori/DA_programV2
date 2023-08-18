from audio_augmentation import AudioAugmentation
from functions import add_noise
import random
import os

class White_Noise(AudioAugmentation):
    def __init__(self):
        self.snr_values = [
            (40, 'easy'),
            (30, 'easy'),
            (25, 'medium'),
            (20, 'medium'),
            (10, 'hard'),
            (0, 'hard')
        ]
        self.type = 1

    def apply(self, input_file, output_file, snr):
        add_noise(input_file, output_file, snr, self.type, noise_file=None)

    def randomize(self):
        random_pair = random.choice(self.snr_values)
        return random_pair
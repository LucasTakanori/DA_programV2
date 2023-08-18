from audio_augmentation import AudioAugmentation
from functions import add_noise
import random
import os

class Brown_Noise(AudioAugmentation):
    def __init__(self):
        self.snr_values = [
            (20, 'easy'),
            (10, 'easy'),
            (0, 'medium'),
            (-10, 'medium'),
            (-20, 'hard'),
            (-30, 'hard')
        ]
        self.type = 3

    def apply(self, input_file, output_file, snr):
        add_noise(input_file, output_file, snr, self.type, noise_file=None)

    def randomize(self):
        random_pair = random.choice(self.snr_values)
        return random_pair
from audio_augmentation import AudioAugmentation
from functions import frequency_mask
import random
class Frequency_Mask(AudioAugmentation):
    def __init__(self, frequency_center):
        self.frequency_values = [
            (100, 'easy'),
            (200, 'easy'),
            (300, 'medium'),
            (400, 'medium'),
            (1000, 'hard'),
            (1500, 'hard')
        ]
        self.frequency_center = frequency_center


    def apply(self, input_file, output_file, width):
        frequency_mask(input_file, output_file, self.frequency_center, width)

    def randomize(self):
        frequency_center = random.choice(self.frequency_values)
        return frequency_center
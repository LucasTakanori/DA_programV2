from audio_augmentation import AudioAugmentation
from functions import frequency_mask
import random
class Frequency_Mask(AudioAugmentation):
    def __init__(self):
        self.width_values = [
            (100.0, 'easy'),
            (200.0, 'easy'),
            (300.0, 'medium'),
            (400.0, 'medium'),
            (1000.0, 'hard'),
            (1500.0, 'hard')
        ]



    def apply(self, input_file, output_file, width):
        frequency_center = 1000 #default value for frequency center
        frequency_mask(input_file, output_file, frequency_center, width)

    def randomize(self):
        width = random.choice(self.width_values)
        return width
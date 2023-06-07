from audio_augmentation import AudioAugmentation
from functions import frequency_mask
import random
class Frequency_Mask(AudioAugmentation):
    def __init__(self, min_frequency_center , max_frequency_center):
        self.min_frequency_center = min_frequency_center
        self.max_frequency_center = max_frequency_center


    def apply(self, input_file, output_file, frequency_center, width):
        frequency_mask(input_file, output_file, frequency_center, width)

    def randomize(self):
        frequency_center = random.uniform(self.min_frequency_center, self.max_frequency_center)
        width = random.uniform(frequency_center/10, frequency_center)
        return frequency_center, width, 

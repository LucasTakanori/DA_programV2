import random
from audio_augmentation import AudioAugmentation
from functions import clipping

class Clipping(AudioAugmentation):
    def __init__(self, min_percentile_threshold, max_percentile_threshold):
        self.min_percentile_threshold = min_percentile_threshold
        self.max_percentile_threshold = max_percentile_threshold

    def apply(self, input_file, output_file, percentile_threshold):
        # Apply clipping augmentation here
        return clipping(input_file, output_file, percentile_threshold)

    def randomize(self):
        percentile_threshold = random.uniform(self.min_percentile_threshold, self.max_percentile_threshold)
        return percentile_threshold,


    
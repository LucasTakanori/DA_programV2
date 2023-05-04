import random
from audio_augmentation import AudioAugmentation
from functions import mp3compression

class Clipping(AudioAugmentation):
    def __init__(self, min_quality, max_quality):
        super().__init__(min_quality, max_quality)

    def apply(self, input_file, output_file, percentile_threshold):
        # Apply clipping augmentation here
        return mp3compression(input_file, output_file, percentile_threshold=10.0)

    def randomize(self):
        percentile_threshold = random.uniform(self.min_quality, self.max_quality)
        return percentile_threshold


    
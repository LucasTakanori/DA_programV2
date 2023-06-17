import random
from audio_augmentation import AudioAugmentation
from functions import mp3compression

class MP3Compression(AudioAugmentation):
    def __init__(self, min_quality, max_quality):
        self.min_quality = min_quality
        self.max_quality = max_quality

    def apply(self, input_file, output_file, quality):
        # Apply clipping augmentation here
        mp3compression(input_file, output_file, quality)

    def randomize(self):
        quality = random.uniform(self.min_quality, self.max_quality)
        return quality,


    
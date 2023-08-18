import random
from audio_augmentation import AudioAugmentation
from functions import mp3compression

class MP3Compression(AudioAugmentation):
    def __init__(self):
        self.compression_values = [
            (40, 'easy'),
            (24, 'easy'),
            (8, 'medium')
        ]

    def apply(self, input_file, output_file, quality):
        # Apply clipping augmentation here
        mp3compression(input_file, output_file, quality)

    def randomize(self):
        random_pair = random.choice(self.compression_values)
        return random_pair



    
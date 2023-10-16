from audio_augmentation import AudioAugmentation
from functions import vtlp
import random
class VTLP(AudioAugmentation):
    def __init__(self):
        self.alpha = [
            (1.2, 'easy'),
            (1.15, 'easy'),
            (1.1, 'easy'),
            (0.9, 'easy'),
            (0.95, 'easy'),
            (0.8, 'easy')
        ]

    def apply(self, input_file, output_file, alpha):
        vtlp(input_file, output_file, alpha=alpha)

    def randomize(self):
        alpha = random.choice(self.alpha)
        return alpha
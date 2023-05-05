from audio_augmentation import AudioAugmentation
from functions import vtlp
import random
class VLTP(AudioAugmentation):
    def __init__(self, min_alpha , max_alpha):
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    def apply(self, input_file, output_file, alpha):
        vtlp(input_file, output_file, alpha=alpha)

    def randomize(self):
        alpha = random.uniform(self.min_alpha, self.max_alpha)
        return alpha,

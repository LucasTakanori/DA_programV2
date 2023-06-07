from audio_augmentation import AudioAugmentation
from functions import add_white_noise
import random
class Add_white_noise(AudioAugmentation):
    def __init__(self, min_snr , max_snr):
        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, input_file, output_file, snr):
        add_white_noise(input_file, output_file, snr)

    def randomize(self):
        snr = random.uniform(self.min_snr, self.max_snr)
        return snr,
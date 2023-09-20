import random
from audio_augmentation import AudioAugmentation
from functions import equalizer

class Equalizer(AudioAugmentation):
    def __init__(self, gains):
        self.gains = gains

        self.type = [
            ('lowpass', 'easy'),
            ('highpass', 'easy')
        ]

        self.lowpass = [0,0,0,0,-40,-40,-40]

        self.highpass = [-40,-40,-40,0,0,0,0]
    
    def apply(self, input_file, output_file, gains):
        # Apply Equalizer augmentation here
        equalizer (input_file, output_file , gains[0], gains[1], gains[2], gains[3], gains[4], gains[5], gains[6])

    def randomize(self):
        type = random.uniform(0,1)
        if type == 0 :
            return self.lowpass,
        else:
            return self.highpass


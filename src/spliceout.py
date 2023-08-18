from audio_augmentation import AudioAugmentation
from functions import splice_out
import random

class Spliceout(AudioAugmentation):
    def __init__(self):
        self.types = 2
        self.times_to_apply = 1
        self.timepercentage_values = [
            (1, 'easy'),
            (5, 'easy'),
            (10, 'medium'),
            (15, 'medium'),
            (20, 'hard'),
            (30, 'hard')
        ]
        

    def apply(self, input_file, output_file, time_percentage):
        # Apply SpliceOut augmentation here
        splice_out(input_file, output_file, self.types, self.times_to_apply , time_percentage)


    def randomize(self):
        random_pair = random.choice(self.timepercentage_values)
        return random_pair
 
    
  
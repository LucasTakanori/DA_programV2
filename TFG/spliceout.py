from audio_augmentation import AudioAugmentation
from functions import splice_out
import random

class spliceout(AudioAugmentation):
    def __init__(self, types, min_time_range, max_time_range, min_times, max_times ,min_snr, max_snr):
        self.types = types
        self.min_time_range = min_time_range
        self.max_time_range = max_time_range
        self.min_times = min_times
        self.max_times = max_times
        self.min_snr = min_snr
        self.max_snr = max_snr

        

    def apply(self, input_file, output_file, types, times_to_apply, time_ranges, snr):
        # Apply SpliceOut augmentation here
        return splice_out(input_file, output_file, types, times_to_apply , time_ranges, snr)

    

    def select_type(self, types):
        # Check if the list of selected types is empty
        if not types:
            raise ValueError("The list of selected types cannot be empty")

        # Filter the selected types to only contain numbers between 1 and 3
        types = [type_ for type_ in types if 1 <= type_ <= 3]

        if not types:
            raise ValueError("The list of selected types can only contain numbers between 1 and 3")

        # Return a random type from the list of selected types
        return random.choice(types)


    def randomize(self):
        snr = random.uniform(self.min_snr, self.max_snr)
        types = self.select_type(self.types)
        times_to_apply = random.randint(self.min_times, self.max_times)
        time_ranges = (self.min_time_range, self.max_time_range)
        return types, times_to_apply, time_ranges, snr,
 
    
  
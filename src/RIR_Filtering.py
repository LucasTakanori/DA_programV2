import random
from audio_augmentation import AudioAugmentation
from functions import RIR_Filtering
import os

class MP3Compression(AudioAugmentation):
    def __init__(self, impulse_folder):
        self.impulse_folder = impulse_folder

    def apply(self, input_file, output_file, impulse_response_txt):
        # Apply clipping augmentation here
        RIR_Filtering(input_file, output_file, impulse_response_txt)

    def randomize(self):
        if(self.impulse_folder is not None):
            # Get a list of all impulse files in the impulse_folder
            impulse_files = [f for f in os.listdir(self.impulse_folder) if os.path.isfile(os.path.join(self.impulse_folder, f))]

            # Select a random audio file
            random_impulse = random.choice(impulse_files)

            # Create the full path of the selected audio file
            impulse_response_txt = os.path.join(self.impulse_folder, random_impulse)

            return impulse_response_txt,
        else:
            raise ValueError("Invalid impulse_folder. Can't be None.")
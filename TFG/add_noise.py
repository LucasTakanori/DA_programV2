from audio_augmentation import AudioAugmentation
from functions import add_noise
import random
import os

class Add_noise(AudioAugmentation):
    def __init__(self, min_snr , max_snr, max_type, min_type, noise_folder):
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.min_type = min_type
        self.max_type = max_type
        self.noise_folder = noise_folder

    def apply(self, input_file, output_file, snr, noise_type, noise_file):
        add_noise(input_file, output_file, snr, noise_type, noise_file)

    def randomize(self):
        snr = random.uniform(self.min_snr, self.max_snr)
        if(self.noise_folder is not None):
            # Get a list of all audio files in the noise_folder
            audio_files = [f for f in os.listdir(self.noise_folder) if os.path.isfile(os.path.join(self.noise_folder, f))]

            # Select a random audio file
            random_audio = random.choice(audio_files)

            # Create the full path of the selected audio file
            random_audio_path = os.path.join(self.noise_folder, random_audio)

            # Return the path of the audio and the snr
            type = None
            return snr, type, random_audio_path,
        else:
            type = random.uniform(self.min_type, self.max_type)
            random_audio_path = None
            return snr, type, random_audio_path,
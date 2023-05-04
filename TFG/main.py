import io
import os
import sys
import random
import shutil
from clipping import Clipping
from vltp import VLTP
from equalizer import Equalizer
from spliceout import spliceout
from mp3compression import MP3Compression
from tqdm import tqdm
from tqdm.utils import _term_move_up
from contextlib import contextmanager

@contextmanager
def redirect_stdout(new_target):
    old_target = sys.stdout
    sys.stdout = new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target


# Gather user input
methods = {
    '1': ('clipping', Clipping(min_percentile_threshold=0, max_percentile_threshold=40)),
    '2': ('vltp', VLTP(min_alpha=0.7, max_alpha=1.4)),
    '3': ('equalizer', Equalizer(gain_min=-40, gain_max=40)),
    '4': ('mp3_compression', MP3Compression(min_quality=0, max_quality=9)),
    '5': ('splice_out', spliceout(types=[1,2,3], min_time_range=0.1, max_time_range=0.4, min_times=1, max_times=8, min_snr=0, max_snr=40))
}

selected_methods_input = input("Enter the augmentation method numbers (1-5) separated by commas: ")

# Validate the selected_methods
try:
    selected_methods = [x.strip() for x in selected_methods_input.split(",") if x.strip().isdigit() and 1 <= int(x.strip()) <= 5]
except ValueError:
    print("Error: Invalid input format. Please enter method numbers (1-5) separated by commas.")
    exit()

input_folder = input("Enter the input folder path: ")

# Validate the input folder
if not os.path.isdir(input_folder):
    print("Error: The input folder does not exist.")
    exit()


output_folder = input("Enter the output folder path: ")

# Validate the output folder
if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)


# Check if the output folder has files, and create a subfolder if necessary
if len(os.listdir(output_folder)) > 0:
    new_subfolder = os.path.join(output_folder, "augmented_files")
    os.makedirs(new_subfolder, exist_ok=True)
    output_folder = new_subfolder

num_augmentations = int(input("Enter the number of times you want to augment the database: "))

# Validate the num_augmentations
try:
    num_augmentations = int(num_augmentations)
    if num_augmentations < 1:
        raise ValueError
except ValueError:
    print("Error: Please enter a valid positive integer for the number of augmentations.")
    exit()

# Prepare the list of audio files
audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]



# Apply the selected augmentation methods randomly
total_augmentations = len(audio_files) * num_augmentations
progress_bar = tqdm(total=total_augmentations, desc="Augmenting", unit="file", position=0, leave=True)

border = "=" * 50
clear_border = _term_move_up() + "\r" + " " * len(border) + "\r"

for audio_file in audio_files:
    input_file = os.path.join(input_folder, audio_file)
    
    # Copy the original file to the output folder
    original_output_file = os.path.join(output_folder, audio_file)
    shutil.copy(input_file, original_output_file)

    # Apply augmentations
    for i in range(1, num_augmentations + 1):
        method_key = random.choice(selected_methods)
        method_name, method_instance = methods[method_key]
        output_file = os.path.join(output_folder, f"{os.path.splitext(audio_file)[0]}_{method_name}_{i}.wav")
        
        # Capture the augmentation method's print output
        message = ""
        with io.StringIO() as buf, redirect_stdout(buf):
            method_instance.apply(input_file, output_file, *method_instance.randomize())
            message = buf.getvalue().strip()

        # Print the message without interfering with the progress bar
        progress_bar.write(clear_border + message)
        progress_bar.write(border)
        
        progress_bar.update(1)  # Update the progress bar

progress_bar.close()  # Close the progress bar when the process is complete
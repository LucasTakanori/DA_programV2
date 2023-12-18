import os
import csv
import random
import shutil
import datetime
from clipping import Clipping
from spliceout import Spliceout
from mp3compression import MP3Compression
from White_noise import White_Noise
from Pink_noise import Pink_Noise
from Brown_noise import Brown_Noise
from frequency_mask import Frequency_Mask
from pitch_shift import Pitch_shift
from vtlp import VTLP
from time_stretch import Time_stretch
from equalizer import Equalizer
from RIR_Filtering import RIR_filtering
from tqdm import tqdm

# Gather user input
methods = {
    '1': ('clipping', Clipping()),
    '2': ('mp3_compression', MP3Compression()),
    '3': ('White_noise', White_Noise()),
    '4': ('Pink_noise', Pink_Noise()),
    '5': ('Brown_noise', Brown_Noise()),
    '6': ('frequency_mask', Frequency_Mask()),
    '7': ('splice_out', Spliceout()),
    '8': ('pitch_shift', Pitch_shift()),
    '9': ('time_stretch', Time_stretch()),
    '10': ('vtlp', VTLP()),

#need to define tags and check how to implement
#    '11': ('equalizer', Equalizer()),              
#    '12': ('RIR_Filtering', RIR_filtering())
}

selected_methods_input = input("Enter the augmentation method numbers (1-10) separated by commas: ") # "1,2,3,4,5,6,7,8,9,10"

# Validate the selected methods
selected_methods = []
for method in selected_methods_input.split(","):
    method = method.strip()
    if method.isdigit() and 1 <= int(method) <= 12:
        selected_methods.append(method)
    else:
        print("Error: Invalid input format. Please enter method numbers (1-12) separated by commas.")
        exit()

tsv_file_path = input("Enter the TSV file path: ") #"../testfiles/RCV/RCV.tsv"

# Validate the TSV file
if not os.path.isfile(tsv_file_path):
    print("Error: The TSV file does not exist.")
    exit()

input_folder = input("Enter the input folder path: ") #"../testfiles/RCV"

# Validate the input folder
if not os.path.isdir(input_folder):
    print("Error: The input folder does not exist.")
    exit()

output_folder = input("Enter the output folder path: ") #"../testfiles/test"

# Validate the output folder
if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)
elif len(os.listdir(output_folder)) > 0:
    # If the output folder exists and is not empty, create a new subfolder
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # Get the current date and time
    new_subfolder = os.path.join(output_folder, f"augmented_files_{timestamp}")
    os.makedirs(new_subfolder, exist_ok=True)
    output_folder = new_subfolder

num_augmentations =  int(input("Enter the number of times you want to augment the database: ")) #2

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
transcripts = {}

with open(tsv_file_path, 'r') as tsv_file:
    tsv_reader = csv.reader(tsv_file, delimiter='\t')
    for row in tsv_reader:
        audio_file, transcript = row
        transcripts[audio_file] = transcript

# Apply the selected augmentation methods randomly
total_augmentations = len(audio_files) * (num_augmentations-1)
progress_bar = tqdm(total=total_augmentations, desc="Augmenting", unit="file", position=0, leave=True)


# Open the output TSV file using csv.writer
with open(os.path.join(output_folder, "output.tsv"), "w", newline="") as tsv_outfile:
    tsv_writer = csv.writer(tsv_outfile, delimiter="\t")
    tsv_writer.writerow(["name_of_the_outputfile", "transcript", "augmentation_method", "randomize_value", "difficulty"])

    # First write all the original files to the TSV
    for audio_file in audio_files:
        input_file = os.path.join(input_folder, audio_file)
        shutil.copyfile(input_file, os.path.join(output_folder, audio_file))
        tsv_writer.writerow([audio_file, transcripts[audio_file]])

    # Then write all the augmentations to the TSV
    for audio_file in audio_files:
        input_file = os.path.join(input_folder, audio_file)
        # Apply augmentations
        for i in range(num_augmentations-1):
            method_key = random.choice(selected_methods)
            method_name, method_instance = methods[method_key]
            output_file = os.path.join(output_folder, f"{os.path.splitext(audio_file)[0]}_{method_name}_{i+1}.wav")

            # Capture the augmentation method's randomize() value
            randomize_value = method_instance.randomize()

            method_instance.apply(input_file, output_file, randomize_value[0])

            # Write the relevant information to the TSV file
            tsv_writer.writerow([os.path.basename(output_file), transcripts[audio_file], method_name, randomize_value[0], randomize_value[1]])

            progress_bar.update(1)  # Update the progress bar

progress_bar.close()  # Close the progress bar when the process is complete
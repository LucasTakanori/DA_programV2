import io
import os
import sys
import random
import shutil
import csv
import wave
import soundfile
import librosa
from clipping import Clipping
from vltp import VLTP
from equalizer import Equalizer
from spliceout import spliceout
from mp3compression import MP3Compression
from tqdm import tqdm
from tqdm.utils import _term_move_up
from contextlib import contextmanager
from pydub import AudioSegment
from PESQ import pesq_from_paths

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
    '5': ('splice_out', spliceout(types=[1,2,3], min_time_range=0.1, max_time_range=0.4, min_times=1, max_times=2, min_snr=0, max_snr=40))
}

selected_methods_input = "1,3,4,5"#input("Enter the augmentation method numbers (1-5) separated by commas: ")

# Validate the selected_methods
try:
    selected_methods = [x.strip() for x in selected_methods_input.split(",") if x.strip().isdigit() and 1 <= int(x.strip()) <= 5]
except ValueError:
    print("Error: Invalid input format. Please enter method numbers (1-5) separated by commas.")
    exit()

tsv_file_path = "../testfiles/CV000/CV000WAV/cleanCV000.tsv"#input("Enter the TSV file path: ")

# Validate the TSV file
if not os.path.isfile(tsv_file_path):
    print("Error: The TSV file does not exist.")
    exit()




input_folder = "../testfiles/CV000/CV000WAV"#input("Enter the input folder path: ")

# Validate the input folder
if not os.path.isdir(input_folder):
    print("Error: The input folder does not exist.")
    exit()


output_folder = "../testfiles/test"#input("Enter the output folder path: ")

# Validate the output folder
if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)

output_tsv_path = "../testfiles/test/output.tsv"  #input("Enter the TSV output file path: ")

if os.path.exists(output_tsv_path):
    os.remove(output_tsv_path)

# Check if the output folder has files, and create a subfolder if necessary
if len(os.listdir(output_folder)) > 0:
    new_subfolder = os.path.join(output_folder, "augmented_files")
    os.makedirs(new_subfolder, exist_ok=True)
    output_folder = new_subfolder

num_augmentations = 2#int(input("Enter the number of times you want to augment the database: "))

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

audio_files = []
transcripts = {}

with open(tsv_file_path, 'r') as tsv_file:
    tsv_reader = csv.reader(tsv_file, delimiter='\t')
    for row in tsv_reader:
        audio_file, transcript = row
        audio_files.append(audio_file)
        transcripts[audio_file] = transcript



# Apply the selected augmentation methods randomly
total_augmentations = len(audio_files) * num_augmentations
progress_bar = tqdm(total=total_augmentations, desc="Augmenting", unit="file", position=0, leave=True)

# Open the output TSV file using csv.writer
with open(output_tsv_path, "a", newline="") as tsv_outfile:
    tsv_writer = csv.writer(tsv_outfile, delimiter="\t")
    tsv_writer.writerow(["name_of_the_outputfile", "transcript", "augmentation_method", "parameter"])

    border = "=" * 150
    clear_border = _term_move_up() + "\r" + " " * len(border) + "\r"

    for audio_file in audio_files:
        input_file = os.path.join(input_folder, audio_file)
        with wave.open(input_file, "rb") as wave_file:
            frame_rate = wave_file.getframerate()

        #print(f"Frame rate: {frame_rate} Hz")
        # Check if the file is MP3 and convert it to WAV with a sample rate of 16k
        if audio_file.lower().endswith(".mp3"):
            mp3_audio = AudioSegment.from_mp3(input_file)
            wav_audio_file = os.path.splitext(audio_file)[0] + ".wav"
            wav_input_file = os.path.join(input_folder, wav_audio_file)
            mp3_audio.set_frame_rate(16000).export(wav_input_file, format="wav")
            os.remove(input_file)  # Delete the original MP3 file
        elif frame_rate != 16000:
            # Load the audio data and downsample it to 16kHz
            y, _ = librosa.load(input_file, sr=16000)
            
            # Save the downsampled audio as a new WAV file
            soundfile.write(input_file, y, 16000, subtype='PCM_16')
            
        else:
            wav_input_file = input_file
            wav_audio_file = audio_file

        # Copy the original file to the output folder
        original_output_file = os.path.join(output_folder, wav_audio_file)
        shutil.copy(wav_input_file, original_output_file)
        tsv_writer.writerow([original_output_file, transcripts[audio_file]])

        # Apply augmentations
        for i in range(1, num_augmentations+1):
            method_key = random.choice(selected_methods)
            method_name, method_instance = methods[method_key]
            output_file = os.path.join(output_folder, f"{os.path.splitext(wav_audio_file)[0]}_{method_name}_{i}.wav")
            
            # Capture the augmentation method's print output
            message = ""
            with io.StringIO() as buf, redirect_stdout(buf):
                params = method_instance.randomize()
                method_instance.apply(wav_input_file, output_file, *params)
                message = buf.getvalue().strip()

            score = pesq_from_paths(wav_input_file, output_file)

            # Write the relevant information to the TSV file
            tsv_writer.writerow([output_file, transcripts[audio_file], method_name, str(params), str(score).replace("tensor(", "").replace(")", "")])

            # Print the message without interfering with the progress bar
            progress_bar.write(clear_border + message)
            progress_bar.write(border)
            
            progress_bar.update(1)  # Update the progress bar

progress_bar.close()  # Close the progress bar when the process is complete

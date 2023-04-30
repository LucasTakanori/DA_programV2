import os
import sys
import subprocess
import mimetypes

def convert_to_wav(input_file, output_file):
    command = f"ffmpeg -i {input_file} -ac 1 -ar 16000 {output_file}"
    subprocess.call(command, shell=True)

def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

    for filename in os.listdir(input_dir):
        input_file = os.path.join(input_dir, filename)

        # Check if the file is an audio file
        mime_type, _ = mimetypes.guess_type(input_file)
        if mime_type is not None and mime_type.startswith("audio"):
            file_root, _ = os.path.splitext(filename)
            output_file = os.path.join(output_dir, f"{file_root}.wav")
            convert_to_wav(input_file, output_file)
        else:
            print(f"Skipping non-audio file: {filename}")

if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    process_directory(input_dir, output_dir)

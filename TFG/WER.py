import jiwer

def load_data(file_path):
    with open(file_path, "r") as f:
        data = [line.strip().split("\t") for line in f]
    return data

def extract_transcriptions(data):
    return [item[1] for item in data]

def calculate_wer(original_transcriptions, augmented_transcriptions):
    return jiwer.wer(original_transcriptions, augmented_transcriptions)

# Load the original and augmented TSV files into two separate lists
original_data = load_data("original.tsv")
augmented_data = load_data("output.tsv")

# Extract transcriptions from the data
original_transcriptions = extract_transcriptions(original_data)
augmented_transcriptions = extract_transcriptions(augmented_data)

# Calculate the Word Error Rate (WER)
error_rate = calculate_wer(original_transcriptions, augmented_transcriptions)

# Print the WER
print("Word Error Rate:", error_rate)

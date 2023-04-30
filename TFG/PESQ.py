import torch
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from scipy.io import wavfile

def read_audio_file(file_path):
    fs, data = wavfile.read(file_path)
    data = torch.tensor(data, dtype=torch.float32)
    return fs, data

def calculate_pesq(fs, ref_audio, deg_audio, mode='wb'):
    return perceptual_evaluation_speech_quality(ref_audio, deg_audio, fs, mode)

def pesq_from_paths(ref_file_path, deg_file_path):
    fs_ref, ref_audio = read_audio_file(ref_file_path)
    fs_deg, deg_audio = read_audio_file(deg_file_path)

    # Ensure the sampling frequencies are the same
    assert fs_ref == fs_deg, "Sampling frequencies must be the same"

    # Choose the mode based on the sampling frequency
    mode = 'wb' if fs_ref == 16000 else 'nb'

    # Calculate PESQ score
    pesq_score = calculate_pesq(fs_ref, ref_audio, deg_audio, mode)
    return pesq_score.item()



ref_file_path = '/home/lucastakanori/DA_programV2/testfiles/UPC_CA_ONA/upc_ca_ona_100000.wav'
deg_file_path = '/home/lucastakanori/DA_programV2/resources/upc_ca_ona_100000vtlp2.wav'
pesq_score = pesq_from_paths(ref_file_path, deg_file_path)
print(f"PESQ score: {pesq_score}")

import torch
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from scipy.io import wavfile
import warnings
# Suppress the specific warning messages
warnings.filterwarnings("ignore", message="To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).")


# Your existing code

def read_audio_file(file_path):
    fs, data = wavfile.read(file_path)
    data = torch.tensor(data, dtype=torch.float32)
    return fs, data


def calculate_pesq(fs, ref_audio, deg_audio, mode):
    # Detach the tensors from the computation graph and move them to the CPU
    ref_audio_detached = ref_audio.detach().cpu()
    deg_audio_detached = deg_audio.detach().cpu()
    
    # Call the perceptual_evaluation_speech_quality function with PyTorch tensors
    return perceptual_evaluation_speech_quality(ref_audio_detached, deg_audio_detached, fs, mode)

def pesq_from_paths(ref_file_path, deg_file_path):
    fs_ref, ref_audio = read_audio_file(ref_file_path)
    fs_deg, deg_audio = read_audio_file(deg_file_path)
    # Ensure the sampling frequencies are the same
    assert fs_ref == fs_deg, "Sampling frequencies must be the same" + str(fs_ref) + str(fs_deg)

    # Choose the mode based on the sampling frequency
    mode = 'wb' if fs_ref == 16000 else 'nb'

    import numpy as np

    if len(ref_audio) > len(deg_audio):
        deg_audio = np.pad(deg_audio, (0, len(ref_audio) - len(deg_audio)))
    elif len(ref_audio) < len(deg_audio):
        deg_audio = deg_audio[:len(ref_audio)]

    # Calculate PESQ score
    pesq_score = calculate_pesq(fs_ref, ref_audio, deg_audio, mode)
    return pesq_score.item()

#ref_file_path = '/home/lucastakanori/DA_programV2/testfiles/tt/common_voice_ca_17368883.wav'
#deg_file_path = '/home/lucastakanori/DA_programV2/testfiles/tt/common_voice_ca_17368883_clipping_2.wav'
#print(pesq_from_paths(ref_file_path,deg_file_path))



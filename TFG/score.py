import librosa
import numpy as np

def calculate_fwSNRseg(audio_path, seg_dur):
    y, sr = librosa.load(audio_path)
    seg_len = int(seg_dur * sr)
    segments = np.array([y[i:i + seg_len] for i in range(0, len(y), seg_len)])
    S_ref = librosa.stft(y)
    S_ref_magnitude = np.abs(S_ref)
    S_ref_power = S_ref_magnitude ** 2
    ref_mean = np.mean(S_ref_power, axis=1)
    snr_segments = []
    for segment in segments:
        S_seg = librosa.stft(segment)
        S_seg_magnitude = np.abs(S_seg)
        S_seg_power = S_seg_magnitude ** 2
        seg_mean = np.mean(S_seg_power, axis=1)
        snr = 10 * np.log10(np.sum(ref_mean) / np.sum(seg_mean))
        snr_segments.append(snr)
    fwSNRseg = np.mean(snr_segments)
    return fwSNRseg

audio_path = '../testfiles/upc_ca_ona_100000.wav'
seg_dur = 0.3 # in seconds
score_in_dBs = calculate_fwSNRseg(audio_path, seg_dur)
print(score_in_dBs)


import numpy as np
from scipy import fft
from scipy.io import wavfile
import soundfile as sf

def frequency_spectrum(x, sf):
    x = x - np.average(x)  # zero-centering
    n = len(x)
    k = np.arange(n)
    tarr = n / float(sf)
    frqarr = k / float(tarr)
    frqarr = frqarr[range(n // 2)]
    x = fft.fft(x) / n
    x = x[range(n // 2)]
    return frqarr, abs(x)

def calculate_fwSNRseg(audio_file):
    data, fs = sf.read(audio_file, dtype='float32')
    y = data[:, 0]  # use the first channel (or take their average, alternatively)
    frq, X = frequency_spectrum(y, fs)
    
    segment_size = 256
    overlap = segment_size // 2
    num_segments = (len(y) - overlap) // (segment_size - overlap)
    
    snr_list = []
    for i in range(num_segments):
        start = i * (segment_size - overlap)
        end = start + segment_size
        segment = y[start:end]
        
        # Calculate the frequency-weighted SNR for this segment
        frq_segment, X_segment = frequency_spectrum(segment, fs)
        signal_power = np.sum(X_segment**2)
        noise_power = np.sum((X - X_segment)**2)
        snr = 10 * np.log10(signal_power / noise_power)
        
        snr_list.append(snr)
    
    # Calculate fwSNRseg
    fwSNRseg = np.mean(snr_list)
    return fwSNRseg

audio_file = 'your_audio_file.wav'
fwSNRseg_score = calculate_fwSNRseg(audio_file)
print(fwSNRseg_score)

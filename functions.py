import librosa
import librosa.display
import librosa.util
import librosa.filters
import librosa.display
from librosa.core import audio

import soundfile as sf

from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

from pydub import AudioSegment

import scipy
from scipy.io.wavfile import write
from scipy import signal
from scipy.signal import butter, lfilter


def load_file(filename):
    return librosa.load(glob(filename)[0])

def calculate_rms(samples):
    """Given a numpy array of audio samples, return its Root Mean Square (RMS)."""
    return np.sqrt(np.mean(np.square(samples)))

def calculate_desired_noise_rms(clean_rms, snr):
    """
    Given the Root Mean Square (RMS) of a clean sound and a desired signal-to-noise ratio (SNR),
    calculate the desired RMS of a noise sound to be mixed in.
    Based on https://github.com/Sato-Kunihiko/audio-SNR/blob/8d2c933b6c0afe6f1203251f4877e7a1068a6130/create_mixed_audio_file.py#L20
    :param clean_rms: Root Mean Square (RMS) - a value between 0.0 and 1.0
    :param snr: Signal-to-Noise (SNR) Ratio in dB - typically somewhere between -20 and 60
    :return:
    """
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a)
    return noise_rms   

def convert_decibels_to_amplitude_ratio(decibels):
    return 10 ** (decibels / 20)

def set_noise(y, noise_rms):
    #amp = convert_decibels_to_amplitude_ratio(decibels)
    #y = np.random.rand(y.shape[0])*amp*2-amp
    noise = np.random.normal(
            0.0, noise_rms, size=y.shape
        )
    return noise


def splice_out(filename, outputfile, type, time_ranges, snr=10):
    y, fs = load_file(filename)

    # Convert time_ranges from seconds to samples
    ranges = [(int(start * fs), int(end * fs)) for start, end in time_ranges]

    if type == 1:
        for start, end in ranges:
            y = np.delete(y, np.s_[start:end])
    elif type == 2:
        for start, end in ranges:
            y[start:end] = 0
    elif type == 3:
        rms = calculate_rms(y)
        noise_rms = calculate_desired_noise_rms(rms, snr)
        for start, end in ranges:
            y[start:end] = set_noise(y[start:end], noise_rms)
    else:
        print("Invalid type input")

    librosa.output.write_wav(outputfile, signal, fs)

#BACK TO WAV?
#test mp3compression
#0->best
#4->perceptual transparency
#6->acceptable
#8->bad
def mp3compression(inputfile,quality=4):
    outputfile=inputfile.split('.')[0]+"_"+str(quality)+".mp3"
    os.system("ffmpeg -y -i " + inputfile +" -codec:a libmp3lame -q:a " + str(quality) + " " +outputfile)


def clipping(filename, outputfile, percentile_threshold=10.0):
    samples, fs = load_file(filename)
    lower_percentile_threshold = percentile_threshold/2
    print(lower_percentile_threshold)
    lower_threshold, upper_threshold = np.percentile(
            samples, [lower_percentile_threshold, 100 - lower_percentile_threshold])
    print(lower_threshold)
    print(upper_threshold)
    samples = np.clip(samples, lower_threshold, upper_threshold)
    pd.Series(samples).plot(figsize=(10, 5), lw=1, title="")
    librosa.output.write_wav(outputfile, samples, fs)

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    filtered = lfilter(b, a, data)
    return filtered

def equalizer (filename, outputfile , gain1=0, gain2=0, gain3=0, gain4=0, gain5=0, gain6=0, gain7=0, gain8=0, gain9=0, gain10=0):
    data, fs = load_file(filename)
    band1 = bandpass_filter(data, 20, 95, fs, order=2)* 10**(gain1/20)
    band2 = bandpass_filter(data, 91, 204, fs, order=3)*10**(gain2/20)
    band3 = bandpass_filter(data, 196, 441, fs, order=3)*10**(gain3/20)
    band4 = bandpass_filter(data, 421, 948, fs, order=3)* 10**(gain4/20)
    band5 = bandpass_filter(data, 909, 2045, fs, order=3)* 10**(gain5/20)
    band6 = bandpass_filter(data, 1957, 4404, fs, order=3)* 10**(gain6/20)
    band7 = bandpass_filter(data, 4216, 9486, fs, order=3)* 10**(gain7/20)
    
    signal = band1 + band2 + band3 + band4 + band5 + band6 + band7 

    # Save output audio to file
    librosa.output.write_wav(outputfile, signal, fs)    

def vtlp_filters(fbank_mx, alpha=1.0, f_high=None):
    """
    Apply vocal tract length perturbation (VTLP) to the filterbank matrix.
    :param fbank_mx: filterbank matrix
    :param alpha: warping factor
    :param f_high: maximum frequency for warping
    :return: warped filterbank matrix
    """
    n_filters, n_fft = fbank_mx.shape
    warped_filters = np.zeros((n_filters, n_fft))
    
    if f_high is None:
        f_high = n_fft / 2
    
    for m in range(n_filters):
        for k in range(n_fft):
            f = (n_fft - 1) * k / (n_fft - 1)
            if f < f_high / 2:
                f_warped = alpha * f
            elif f < f_high:
                f_warped = alpha * f + (1 - alpha) * (f_high / 2)
            else:
                f_warped = f
            
            k_warped = int(n_fft * f_warped / (n_fft - 1))
            if k_warped < n_fft:
                warped_filters[m, k_warped] += fbank_mx[m, k]
    
    return warped_filters



def vtlp(input_wav_file, output_wav_file, alpha=1.0, f_high=None):
    """
    Apply vocal tract length perturbation (VTLP) to a .wav file.
    :param input_wav_file: input .wav file
    :param output_wav_file: output .wav file
    :param alpha: warping factor
    :param f_high: maximum frequency for warping
    """
    # Load the input .wav file
    y, sr = librosa.load(input_wav_file)
    
    # Compute the spectrogram
    n_fft = 2048
    hop_length = 512
    win_length = n_fft
    window = scipy.signal.windows.hann(win_length, sym=False)
    
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    
    # Compute the filterbank matrix
    n_mels = 256
    fbank_mx = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    
    # Apply VTLP to the filterbank matrix
    warped_filters = vtlp_filters(fbank_mx, alpha=alpha, f_high=f_high)
    
    # Compute the mel spectrogram using the warped filterbank matrix
    S = np.dot(warped_filters, np.abs(D))
    
    # Invert the mel spectrogram to audio
    y_hat = librosa.feature.inverse.mel_to_audio(S,
                                                 sr=sr,
                                                 n_fft=n_fft,
                                                 hop_length=hop_length,
                                                 win_length=win_length,
                                                 window=window,
                                                 power=1,
                                                 n_iter=128,
                                                 length=len(y))
    
    # Save the output .wav file
    write(output_wav_file, sr, y_hat)

def mp3towav(input,output):
    os.system("ffmpeg -i " + input + " -ar 44k " + output)

def visualize_signals(original, transformed):
    y, fs1 = load_file(original)
    y_tr, fs2 = load_file(transformed)

    time = pd.Series(range(len(y))) / fs1
    time1 = pd.Series(range(len(y_tr))) / fs2

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(time, y, '-r', label=r"$Original (t)$")
    plt.title('Original Signal')
    plt.xlabel('time[s]')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(time1, y_tr, '-b',label=r"$Transformed amplitude(t)$")
    plt.title('Transformed Signal ')
    plt.xlabel('time[s]')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()
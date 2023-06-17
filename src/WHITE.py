import os
import glob
import librosa
import numpy as np
import soundfile as sf
def add_white_noise(input_filename, output_filename, desired_snr):
    # Read the input audio file
    signal, sr = librosa.load(input_filename, sr=None)

    # Remove silence from the start and end of the signal
    trimmed_signal, _ = librosa.effects.trim(signal)

    # Generate white noise
    noise = np.random.normal(0, 1, trimmed_signal.shape)

    # Calculate the signal and noise power
    signal_power = np.sum(trimmed_signal ** 2) / trimmed_signal.size
    noise_power = np.sum(noise ** 2) / noise.size

    # Calculate the scaling factor for the desired SNR level
    scaling_factor = np.sqrt((signal_power / noise_power) * 10 ** (-desired_snr / 10))

    # Scale the noise
    noise_scaled = noise * scaling_factor

    # Add the scaled white noise to the trimmed signal
    noisy_signal = trimmed_signal + noise_scaled

    # Save the modified audio signal to the output file
    sf.write(output_filename, noisy_signal, sr)
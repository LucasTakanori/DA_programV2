#apply_vtlp function that identifies the formants of the input audio signal and applies the VTLP transform using those formants:

import librosa
import numpy as np
#import sounddevice as sd

def apply_vtlp(signal, vtlp_factor):
    # Calculate the mel spectrogram of the input signal
    mel_spec = librosa.feature.melspectrogram(signal, sr=signal.fs, n_fft=2048, hop_length=512, n_mels=128)

    # Convert the mel spectrogram to dB units
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Find the peaks in the mel spectrogram
    peak_frequencies, _ = librosa.core.peak_pick(mel_spec_db, 3, 3, 3, 5, 0.5, 10)

    # Sort the peaks by frequency
    sorted_peaks = np.sort(peak_frequencies)

    # Calculate the new formant frequencies based on the VTL factor and the sorted peaks
    new_formants = sorted_peaks * vtlp_factor

    # Calculate the bandwidths of the formants
    bw = new_formants / 2.0

    # Define the center frequencies and bandwidths of the formants
    formants = np.stack((new_formants, bw), axis=1)

    # Apply a bandpass filter to the signal at the frequencies of the formants
    filtered_signal = np.zeros_like(signal)
    for i in range(formants.shape[0]):
        center_freq = formants[i][0]
        bandwidth = formants[i][1]
        b, a = librosa.core.audio.util.band_pass(center_freq - bandwidth, center_freq + bandwidth, signal.fs)
        filtered_signal += librosa.core.audio.util.pad_center(librosa.core.audio.util.filtfilt(b, a, signal), len(signal))

    return filtered_signal




# Load an audio file
y, sr = librosa.load(librosa.example('nutcracker'))

# Define the VTL factor
vtlp_factor = 1.2

# Apply the VTL factor to the signal
vtlp_signal = apply_vtlp(y, vtlp_factor)

# Play the original signal
#sd.play(y, sr)
#sd.wait()

# Play the signal with the VTL factor applied
#sd.play(vtlp_signal, sr)
#sd.wait()


# Play the original signal
librosa.output.write_wav('original.wav', y, sr)

# Play the signal with the VTL factor applied
librosa.output.write_wav('vtlp.wav', vtlp_signal, sr)

#This function takes as input an audio signal and a VTLP factor. It first calculates the mel spectrogram of the input signal using librosa.feature.melspectrogram, and then identifies the three highest peaks in the mel spectrogram using librosa.core.peak_pick.
#The function then sorts the peak frequencies by frequency, and calculates the new formant frequencies based on the VTLP factor and the sorted peaks. The bandwidths of the formants are calculated as half of the new formant frequencies.
#Finally, the function applies a bandpass filter to the signal at the frequencies of the formants using librosa.core.audio.util.band_pass, and returns the filtered signal.
#Note that this is just a simple example implementation of formant-based VTLP, and there are many other techniques and parameters that can be used depending on the specific application.

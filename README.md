# DA_programV2

DA_programV2 is a data augmentation program that includes several functions for audio processing.

## Installation

To install DA_programV2, follow these steps:

1. Clone the repository: `git clone https://github.com/LucasTakanori/DA_programV2`
2. Install the required libraries and dependencies: `pip install -r requirements.txt`
3. Install ffmpeg `http://ffmpeg.org/download.html`

## Usage

To use DA_programV2, import the necessary functions from the program. The main functions include:

### `splice_out(filename, outputfile, type, time_ranges, snr=10)`
-  This function applies splicing to an audio file and saves the spliced audio to a new file. 

    The `filename` parameter is the path to the input audio file. 
    
    The `outputfile` parameter is the path to the output audio file. 
    
    The `type` parameter controls the type of splicing applied to the audio and can be 1 (delete), 2 (silence), or 3 (noise). 
    
    The `time_ranges` parameter is a list of tuples specifying the start and end times (in seconds) of the regions to be spliced. 
    
    The `snr` parameter controls the signal-to-noise ratio when `type` is set to 3 (noise). The default value for `snr` is 10. 
    
    The function applies the specified type of splicing to the specified time ranges in the audio file and saves the spliced audio signal to the specified output file using the `librosa.output.write_wav` function.

### `mp3compression(inputfile, outputfile, quality=4)`
-   This function compresses an audio file using the MP3 codec. 

    The `inputfile` parameter is the path to the input audio file. 
    
    The `quality` parameter controls the quality of the compressed audio and can range from 0 (best) to 9 (worst). The default value is 4, which provides perceptual transparency. 
    
    The function creates a new compressed audio file with the same name as the input file but with an additional suffix indicating the quality level and an ".mp3" extension.

### `vtlp(input_wav_file, outputfile, alpha=1.0, f_high=None)`
-   This function applies Variable Track Length Perturbation to an audio file and saves the spliced audio to a new file. 

    This function loads the input .wav file using `librosa.load`, computes the spectrogram using `librosa.stft`, and computes the filterbank matrix using librosa.filters.mel. It then applies VTLP to the filterbank matrix using the `vtlp_filter` function. The mel spectrogram is computed using the warped filterbank matrix and inverted to audio using `librosa.feature.inverse.mel_to_audio`. Finally, the output .wav file is saved using `scipy.io.wavfile.write`.

### `equalizer(filename, outputfile, gain1=0, gain2=0, gain3=0, gain4=0, gain5=0, gain6=0, gain7=0)`
-   This function applies a 7-band equalizer to an audio file and saves the equalized audio to a new file. 

    The `filename` parameter is the path to the input audio file. 
    
    The `outputfile` parameter is the path to the output audio file. 
    
    The `gain1` to `gain7` parameters control the gain applied to each of the 7 frequency bands. The default value for each gain parameter is 0. 
    
    The function applies a bandpass filter to each frequency band and multiplies the filtered signal by the corresponding gain value. The function then sums the signals from all 7 frequency bands to produce the equalized audio signal. The function saves the equalized audio signal to the specified output file using the `librosa.output.write_wav` function.

### `clipping(filename, outputfile, percentile_threshold=20.0)`
-   This function applies clipping to an audio file. 

    The `filename` parameter is the path to the input audio file. 
    
    The `percentile_threshold` parameter controls the amount of clipping applied to the audio and can range from 0 to 100. The default value is 20.0. 
    
    The function calculates the lower and upper thresholds for clipping based on the specified percentile threshold. The lower threshold is calculated as the value below which a given percentage of the audio samples fall, and the upper threshold is calculated as the value above which a given percentage of the audio samples fall. The function then applies clipping to the audio samples by setting any samples below the lower threshold to the lower threshold value and any samples above the upper threshold to the upper threshold value. the output .wav file is saved using `librosa.output.write_wav`.

Here are some examples of how to use these functions:

## Original audio

[Listen to audio](resources/upc_ca_ona_100000.wav)

![Alt text](resources/upc_ca_ona_100000.png)
   
`splice_out: max ranges: 8, max duration: 400ms, snr max: 40db`
        
        splice_out("../testfiles/upc_ca_ona_100000.wav","../resources/upc_ca_ona_100000SPLICE1.wav",1,[(6, 6.2),(7.5 , 7.7),(10, 10.2)])

[Listen to audio](resources/upc_ca_ona_100000SPLICE1.wav)

![Alt text](resources/SPLICE1.png)

        splice_out("../testfiles/upc_ca_ona_100000.wav","../resources/upc_ca_ona_100000SPLICE2.wav",2,[(6, 6.2),(7.5 , 7.7),(10, 10.2)])

[Listen to audio](resources/upc_ca_ona_100000SPLICE2.wav)
![Alt text](resources/SPLICE2.png)

        splice_out("../testfiles/upc_ca_ona_100000.wav","../resources/upc_ca_ona_100000SPLICE3.wav",3,[(6, 6.2),(7.5 , 7.7),(10, 10.2)])

[Listen to audio](resources/upc_ca_ona_100000SPLICE3.wav)
![Alt text](resources/SPLICE3.png)

`mp3compression: 0 (best) to 9 (worst) quality`

        mp3compression("../testfiles/upc_ca_ona_100000.wav","../resources/upc_ca_ona_100000mp3.wav", 8)

[Listen to audio](resources/upc_ca_ona_100000mp3.wav)
![Alt text](resources/mp3.png)

`vtlp: min alpha: 0.8, max alpha: 1.2`

        vtlp("../testfiles/upc_ca_ona_100000.wav","../resources/upc_ca_ona_100000vtlp.wav",1.2)

[Listen to audio](resources/upc_ca_ona_100000vtlp.wav)
![Alt text](resources/vtlp.png)

`equalizer: min gain: -15db, max gain 15db`

        equalizer("../testfiles/upc_ca_ona_100000.wav","../resources/upc_ca_ona_100000EQ.wav",-15,-15,-15,-15,-15,-15,-15)

[Listen to audio](resources/upc_ca_ona_100000EQ.wav)
![Alt text](resources/EQ.png)

`clipping: min clipping: 0%, max clipping: 40%`

        clipping("../testfiles/upc_ca_ona_100000.wav","../resources/upc_ca_ona_100000Clipping.wav",5)

[Listen to audio](resources/upc_ca_ona_100000Clipping.wav)
![Alt text](resources/Clipping.png)


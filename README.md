# DA_programV2

DA_programV2 is a data augmentation program that includes several functions for audio processing.

## Installation

This software was developed in Ubuntu 22.04.2 LTS and uses Python 3.10.6.

To install DA_programV2, follow these steps:

1. Clone the repository: `git clone https://github.com/LucasTakanori/DA_programV2`
2. Install the required libraries and dependencies: `pip install -r requirements.txt`
3. Install ffmpeg from `http://ffmpeg.org/download.html`

## Usage

To use DA_programV2, run the main.py script from the src folder. 

First the program will ask for the agumentation methods that we want to use separated by comas.

Second the program will ask for an input tsv file with the original database namefiles and transcriptions.

Third the program will ask for an input folder with the files to augment.

Fourth the program will ask for an output folder where the augmented files will be stored.

Finally the program will ask for the number of times that the database will be augmented.
Here's an example of usage:

```bash
lucastakanori@DESKTOP-5CD06JF:~/DA_programV2/src$ python3 main.py
Enter the augmentation method numbers (1-12) separated by commas: 1,2,3,4
Enter the TSV file path: ../testfiles/RCV/RCV.tsv
Enter the input folder path: ../testfiles/RCV
Enter the output folder path: ../testfiles/test
Enter the number of times you want to augment the database: 2
```

If you need to apply the script multiple times it's faster to hardcode the preferred parameters into the script file.
### EXAMPLE TSV OUTPUT WITH 10 FILES NUM AUGMENTATION = 2

| name_of_the_outputfile                         | transcript                                                                                                              | augmentation_method | randomize_value | difficulty |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------- | --------------- | ---------- |
| common_voice_ca_17367931.wav                   | Què, doncs?                                                                                                             |                     |                 |            |
| common_voice_ca_17367675.wav                   | La durada dels projectes no pot sobrepassar els tres anys, des de la data de la sol·licitud.                            |                     |                 |            |
| common_voice_ca_17367896.wav                   | Grapa.                                                                                                                  |                     |                 |            |
| common_voice_ca_17367777.wav                   | Somrigué.                                                                                                               |                     |                 |            |
| common_voice_ca_17367650.wav                   | La distància màxima des del marge fins a la darrera fila de plantes conreades no podrà ser superior a dos-cents metres. |                     |                 |            |
| common_voice_ca_17367588.wav                   | Passa alguna cosa?                                                                                                      |                     |                 |            |
| common_voice_ca_17367950.wav                   | El recordem amb catorze pensaments seus.                                                                                |                     |                 |            |
| common_voice_ca_17367585.wav                   | Tots.                                                                                                                   |                     |                 |            |
| common_voice_ca_17367674.wav                   | Igualment fan rotació de manera consecutiva en horari de matí, tarda i nit.                                             |                     |                 |            |
| common_voice_ca_17367922.wav                   | Oh, Senyor!                                                                                                             |                     |                 |            |
| common_voice_ca_17367931_clipping_1.wav        | Què, doncs?                                                                                                             | clipping            | 20.0            | medium     |
| common_voice_ca_17367675_pitch_shift_1.wav     | La durada dels projectes no pot sobrepassar els tres anys, des de la data de la sol·licitud.                            | pitch_shift         | 0.8             | easy       |
| common_voice_ca_17367896_mp3_compression_1.wav | Grapa.                                                                                                                  | mp3_compression     | 8               | medium     |
| common_voice_ca_17367777_frequency_mask_1.wav  | Somrigué.                                                                                                               | frequency_mask      | 1000.0          | hard       |
| common_voice_ca_17367650_White_noise_1.wav     | La distància màxima des del marge fins a la darrera fila de plantes conreades no podrà ser superior a dos-cents metres. | White_noise         | 40              | easy       |
| common_voice_ca_17367588_Pink_noise_1.wav      | Passa alguna cosa?                                                                                                      | Pink_noise          | 0               | hard       |
| common_voice_ca_17367950_clipping_1.wav        | El recordem amb catorze pensaments seus.                                                                                | clipping            | 1.0             | easy       |
| common_voice_ca_17367585_time_stretch_1.wav    | Tots.                                                                                                                   | time_stretch        | 0.95            | easy       |
| common_voice_ca_17367674_pitch_shift_1.wav     | Igualment fan rotació de manera consecutiva en horari de matí, tarda i nit.                                             | pitch_shift         | 1.2             | easy       |
| common_voice_ca_17367922_vtlp_1.wav            | Oh, Senyor!                                                                                                             | vtlp                | 0.8             | easy       |
|                                                |                                                                                                                         |                     |                 |            |


## The data augmentation functions implemented in the code are the following:

`Clipping`: Clips audio samples to specified minimum and maximum values

`Equalizer`: Adjusts the volume of 7 frequency bands

`Frequency Masking`: Applies a frequency mask

`Mp3compression`: Compresses the audio to lower the quality

`Noise Adding`:  Adds noise to the audio samples

`Pitch Shifting`: Shifts the pitch up or down without changing the tempo

`RIR Filtering`: Convolves the audio with a randomly chosen impulse response

`Splice Out`: Applies different types of time masking

`Time Stretching`: Changes the speed without changing the pitch

`VTLP`: Applies Vocal Tract Lenght Perturbation to an audio signal


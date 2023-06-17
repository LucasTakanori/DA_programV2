#input csv amb path, text --> N arxius a obrir
#Demanar P metodes a fer servir
#Demanar M cops a agumentar (aplicar Mx(N-1) cops)
#importar P metodes
#Llista N arxius-> asignar aleatoriament 1 metode 
#amb parametres aleatoris a 1 arxiu repetir fins arribar a M x(N-1) -> 
#->llista Mx(N-1) operacions
#ordenar llista per nom
#aplicar arxiu per arxiu K augmentacions (guardar nom arxiu nou, tipus augmentació, 
# label dificultat, parametres utilitzats)
#repetir per N arxius 
#guardar en csv NXM --path --text --score --tipus --parametres 

from clipping import Clipping
from vltp import VLTP
from equalizer import Equalizer
from spliceout import spliceout
from mp3compression import MP3Compression

input_file = "../testfiles/UPC_CA_ONA_WAV/upc_ca_ona_100000.wav"
output_file = "../testfiles/CLASSTEST/upc_ca_ona_100000"
extension = ".wav"
types = [1, 2, 3]

# Initialize augmentation objects
clipping = Clipping(min_percentile_threshold=0, max_percentile_threshold=40)
vltp = VLTP(min_alpha=0.7, max_alpha=1.4)
equalizer = Equalizer(gain_min=-40, gain_max=40)
mp3_compression = MP3Compression(min_quality=0, max_quality=9)
splice_out = spliceout(types, min_time_range=0.1, max_time_range=0.4, min_times=1, max_times=8 ,min_snr=0, max_snr=40)

# Apply augmentation
clipping.apply(input_file, output_file+"CLIPPING"+extension, clipping.randomize())
vltp.apply(input_file, output_file+"vltp"+extension, vltp.randomize())
equalizer.apply(input_file, output_file+"equalizer"+extension, equalizer.randomize())
mp3_compression.apply(input_file, output_file+"mp3"+extension, mp3_compression.randomize())
splice_out.apply(input_file, output_file+"splice"+extension, *splice_out.randomize())


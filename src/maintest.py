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
from spliceout import Spliceout
from mp3compression import MP3Compression
from White_noise import White_Noise
from Pink_noise import Pink_Noise
from Brown_noise import Brown_Noise
from frequency_mask import Frequency_Mask


input_file = "upc_ca_ona_100000.wav"
output_file = "test/upc_ca_ona_100000"
extension = ".wav"
types = [1, 2, 3]

# Initialize augmentation objects
clipping = Clipping()
mp3_compression = MP3Compression()
White_noise = White_Noise()
Pink_noise = Pink_Noise()
Brown_noise = Brown_Noise()
frequency_mask = Frequency_Mask()
splice_out = Spliceout()

# Apply augmentation

clipping.apply(input_file, output_file+"CLIPPING"+extension, clipping.randomize()[0])
mp3_compression.apply(input_file, output_file+"mp3"+extension, mp3_compression.randomize()[0])
White_noise.apply(input_file, output_file+"WHITE"+extension, White_noise.randomize()[0])
Pink_noise.apply(input_file, output_file+"PINK"+extension, Pink_noise.randomize()[0])
Brown_noise.apply(input_file, output_file+"BROWN"+extension, Brown_noise.randomize()[0])
frequency_mask.apply(input_file, output_file+"FREQUENCY"+extension,frequency_mask.randomize()[0])
splice_out.apply(input_file, output_file+"splice"+extension, splice_out.randomize()[0])



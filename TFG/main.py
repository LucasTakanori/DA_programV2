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

input_file = "input.wav"
output_file = "output.wav"
selected_types = [1, 2, 3]

# Initialize augmentation objects
clipping = Clipping(min_param=0.1, max_param=0.5)
vltp = VLTP(min_param=0.1, max_param=0.5)
equalizer = Equalizer(min_param=0.1, max_param=0.5)
splice_out = spliceout(min_param=0.1, max_param=0.5)

# Apply augmentation
clipping.apply(input_file, output_file)
vltp.apply(input_file, output_file)
equalizer.apply(input_file, output_file)
splice_out.apply(input_file, output_file)

# Randomize parameters
clipping.randomize()
vltp.randomize()
equalizer.randomize()
splice_out.randomize()

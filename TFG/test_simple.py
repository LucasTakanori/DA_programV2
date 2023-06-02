#!/usr/bin/env python3

import json
import wave
import sys
import os

from vosk import Model, KaldiRecognizer, SetLogLevel

# You can set log level to -1 to disable debug messages
SetLogLevel(0)

wf = wave.open(sys.argv[1], "rb")
if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
    print("Audio file must be WAV format mono PCM.")
    sys.exit(1)

model = Model(lang="ca")
rec = KaldiRecognizer(model, wf.getframerate())
rec.SetWords(True)
rec.SetPartialWords(True)
last = None
while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        #print(rec.Result())
        last = rec.Result()
        print(last)
        print("pi")
    else:
        print(rec.PartialResult())
        print("pu")
        
result_json = rec.FinalResult()
result_dict = json.loads(result_json)
print(rec.FinalResult())
#print(last)
if(result_dict["text"] == ""):
    result_dict = json.loads(last)        
#print("Parsed JSON:", result_dict)

#results = result_dict["result"]
#conf_scores = [item["conf"] for item in results]
#average_conf = sum(conf_scores) / len(conf_scores)
#print(average_conf)
# Function to save the output to a TSV file
def save_to_tsv(filename, transcription):#, avg_confidence):
    with open(filename, "w") as f:
        f.write(f"{os.path.basename(sys.argv[1])}\t{transcription}")#\t{avg_confidence}")

# Save the output to a TSV file
save_to_tsv("output.tsv", result_dict["text"])#, average_conf)

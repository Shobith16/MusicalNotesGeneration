import librosa
from audio_to_midi.sound_to_midi.monophonic import wave_to_midi

def audio_to_midi(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=None)
        print("Audio file loaded!")
        midi = wave_to_midi(y, sr)
        print("Conversion finished!")

        outputfile = audio_file[:-4] + ".mid"
        with open(outputfile, 'wb') as f:
            midi.writeFile(f)
        print("MIDI file saved:", outputfile)
        return outputfile
    except Exception as e:
        print("Error occurred during audio to MIDI conversion:", e)
        return None

# Function to predict MIDI file
def pred_midifile(audio_file):
    try:
        midifile = audio_to_midi(audio_file)
        if midifile:
            return midifile
        else:
            return "Error converting audio to MIDI"
    except Exception as e:
        print("Error occurred during MIDI prediction:", e)
        return None

# # Example usage
# audio_file_path = "D:/Final/Musical_Notes/audio/tum-hii-ho.wav"
# predicted_midifile = pred_midifile(audio_file_path)
# if predicted_midifile:
#     print("Predicted MIDI file path:", predicted_midifile)
# else:
#     print("Error predicting MIDI file")

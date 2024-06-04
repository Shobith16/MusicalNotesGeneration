import os
import librosa
import numpy as np

def detect_silent_audio(audio_file, threshold_db=-50, frame_length=2048, hop_length=512):
    # Load audio file
    y, sr = librosa.load(audio_file)
    
    # Calculate energy of each frame
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Convert energy to decibels
    energy_db = librosa.amplitude_to_db(energy)
    
    # Find frames below the threshold
    silent_frames = np.where(energy_db < threshold_db)[0]
    
    return silent_frames, len(energy_db)

def find_empty(file_path):
    audiofile = []
    path = 'separatedfiles/'
    
    for file in file_path:
        print(file)
        fileloc = os.path.join(path, file)
        silent_frames, total_frames = detect_silent_audio(fileloc)

        if len(silent_frames) == total_frames:
            print(f"Detected {len(silent_frames)} silent frames. Entire file is silent:", file)
        elif file.endswith('.wav'):
            print("This file has audio:", file)
            audiofile.append(file)
    
    return audiofile
def findempty(file):
    return find_empty(file)
# # Example usage
# file_path = ['other.wav', 'bass.wav', 'drums.wav', 'vocals.wav']
# audio_files_with_audio = find_empty(file_path)
# print("Files with audio:", audio_files_with_audio)

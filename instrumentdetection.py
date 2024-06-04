import numpy as np
import librosa
# import os
import joblib
from sklearn.ensemble import RandomForestClassifier


# Function to extract features from audio files
def extract_features(audio_file, max_length=100):
    # Load audio file
    y, sr = librosa.load(audio_file, mono=True)
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # Calculate the number of columns to pad or truncate
    pad_width = max_length - mfccs.shape[1]
    if pad_width > 0:
        # Pad the MFCC matrix with zeros
        padded_mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    else:
        # Truncate the MFCC matrix if pad_width is negative
        padded_mfccs = mfccs[:, :max_length]
    # Flatten the MFCC matrix into one dimension
    flat_mfccs = np.ravel(padded_mfccs)
    return flat_mfccs



# Function to compare input audio as musical instrument or non-musical sound
def compare_audio(model_path, input_audio, threshold=0.7):
    # Load the saved model
    model = joblib.load(model_path)
    
    # Extract features from input_audio
    input_features = extract_features(input_audio)
    
    # Predict label using the loaded model
    predicted_label = model.predict([input_features])[0]
    
    # Get the probability of the predicted label
    predicted_prob = model.predict_proba([input_features])[0][model.classes_.tolist().index(predicted_label)]
    
    # Print the result
    if predicted_prob >= threshold:
        if predicted_label == "musical_instruments":
            print("The input audio is classified as a musical instrument.")
            return "musical_instrument"
        else:
            print("The input audio is classified as a non-musical sound (e.g., dog barking).")
            return "not_musical_instrument"
    else:
        print("Unable to classify the input audio.")
        return "Unable to classify the input audio."

# Example usage
def detect_vaildfile(filename):
    model_path="audio_model.pkl"
    input_audio_file = filename
    return compare_audio(model_path, input_audio_file, threshold=0.7)


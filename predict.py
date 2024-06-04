import librosa
import numpy as np
import tensorflow as tf
from tensorflow.image import resize
from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dense, Input
# from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
import os

# Load the saved model
model = load_model('improved_audio_classification_model.h5')

# Define the target shape for input spectrograms
target_shape = (128, 128)

# Define your class labels
classes = ['Sound_Drum', 'Sound_Guitar', 'Sound_Piano', 'Sound_Violin']

# Function to preprocess and classify an audio file
def test_audio(file_path, model):
    # Load and preprocess the audio file
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    
    # Noise reduction
    # audio_data = librosa.effects.harmonic(audio_data)
    # audio_data = librosa.effects.percussive(audio_data)
    
    # # Normalization
    # audio_data = (audio_data - np.min(audio_data)) / (np.max(audio_data) - np.min(audio_data))
    
    # Compute Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    
    # Resize and reshape the Mel spectrogram
    mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
    mel_spectrogram = tf.reshape(mel_spectrogram, (1,) + target_shape + (1,))
    
    # Make predictions
    predictions = model.predict(mel_spectrogram)
    
    # Get the class probabilities
    class_probabilities = predictions[0]
    
    # Get the predicted class index
    predicted_class_index = np.argmax(class_probabilities)

    # Display results for all classes
    for i, class_label in enumerate(classes):
        probability = class_probabilities[i]
        print(f'Class: {class_label}, Probability: {probability:.4f}')
    
    # Calculate and display the predicted class and accuracy
    predicted_class = classes[predicted_class_index]
    accuracy = class_probabilities[predicted_class_index]
    print(f'The audio is classified as: {predicted_class}')
    print(f'Accuracy: {accuracy:.4f}')
    
    return predicted_class

# Function to predict instrument
def predict_instrument(audio_file):
    predicted_instrument = test_audio(audio_file, model)
    print(predicted_instrument)
    return predicted_instrument

# Example usage
# predict_instrument("music/acoustic-guitar-loop-f-91bpm-132687.mp3")
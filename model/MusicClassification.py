from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.image import resize
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
# Define the input shape based on the size of the spectrogram
target_shape = (128, 128)  # Modify this based on your target shape

# Load and preprocess audio data
def load_and_preprocess_data(train_dir, classes, target_shape=(128, 128)):
    data = []
    labels = []
    
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(train_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(class_dir, filename)
                audio_data, sample_rate = librosa.load(file_path, sr=None)
                # Perform preprocessing (e.g., convert to Mel spectrogram and resize)
                mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
                mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
                data.append(mel_spectrogram)
                # print(i)
                labels.append(i)
    
    return np.array(data), np.array(labels)

# Define the directories and classes
train_dir = 'D:/Final/Musical_Notes/model/dataset/'
train_df = pd.read_csv('model/output.csv')
classes = sorted(train_df['Class'].unique().tolist())
print(classes)
# Load and preprocess data
data, labels = load_and_preprocess_data(train_dir, classes, target_shape)
labels = to_categorical(labels, num_classes=len(classes))  # Convert labels to one-hot encoding
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Model architecture with batch normalization and dropout
input_layer = Input(shape=target_shape + (1,))  # Add the channel dimension (1 for grayscale)
x = Conv2D(16, (3, 3), activation='relu')(input_layer) 
# x = BatchNormalization()(x)      #filtering the unwanted pxls in the image using backpropagation 
x = MaxPooling2D((2, 2))(x)                                #dence filter pics the most wanted part in the filter image                                     #handle the fluctuation during backpropagation to avoid model failure
x = Conv2D(32, (3, 3), activation='relu')(x)
# x = BatchNormalization()(x)  
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
# x = BatchNormalization()(x)  
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
# x = BatchNormalization()(x)  
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)                                           #coverting multi dimention to single dimention (rolling)
x = Dense(128, activation='relu')(x)                       #predict with the 128 nerons
x = Dropout(0.4)(x)                                         #removes random 40%
output_layer = Dense(len(classes), activation='softmax')(x)
model = Model(input_layer, output_layer)



lr_scheduler = ReduceLROnPlateau(factor=0.1, patience=3, min_lr=1e-6)
early_stopping = EarlyStopping(patience=15, restore_best_weights=True)

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with callbacks
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), callbacks=[lr_scheduler, early_stopping])
# # Plot training history
# plt.figure(figsize=(10, 5))

# # Plot training & validation accuracy values
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')

# # Plot training & validation loss values
# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')

# plt.tight_layout()
# plt.show()
# Evaluate the model
test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(test_accuracy[1])

# Save the model
model.save('improved_audio_classification_model.h5')
# Evaluate the model and get predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Generate classification report
classification_rep = classification_report(y_true_classes, y_pred_classes, labels=np.arange(len(classes)), target_names=classes, zero_division=1)
print(classification_rep)

---

# Automatic Music Transcription to Musical Notes

## Project Overview

This project aims to automate the transcription of music into musical notes. The process involves separating audio tracks into individual components (such as drums, vocals, bass, etc.), classifying the musical instruments, converting the sound to MIDI format, and extracting pitch information to display the musical notes.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- Audio track separation using the HDemucs model.
- Musical instrument classification (guitar, drums, piano, violin).
- Sound to MIDI conversion.
- Extraction and display of musical notes from MIDI files.

## Technologies Used

- **Python**: Programming language.
- **HDemucs**: Deep learning model for audio source separation.
- **Librosa**: Python library for analyzing audio and music.
- **Pandas**: Data manipulation and analysis library.
- **TensorFlow/Keras**: Frameworks for building and training machine learning models.
- **MIDIUtil**: Library for creating and manipulating MIDI files.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/Shobith16/MusicalNotesGeneration/
   cd MusicalNotesGeneration
   ```

2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. **Audio Track Separation**:
   - Use the HDemucs model to separate the audio tracks.
   ```python
   from demucs.separate import load_track, separate

   audio_file = 'path/to/musicfile'
   separated_files = mixed_sep(audio_file)
   ```

2. **Instrument Classification**:
   - Train the model on the provided dataset or use a pre-trained model for classifying instruments.
   ```python
   # Load your model and classify the instrument
   model = load_model('path/to/model')
   prediction =predict_instrument("path/to/musicfile")
   ```

3. **Sound to MIDI Conversion**:
   - Convert the separated and classified audio tracks to MIDI format.
   ```python
   from musictomid import pred_midifile

    mid_file = pred_midifile(audio_file)
   ```

4. **Extract and Display Notes**:
   - Extract pitch information from the MIDI file and display the musical notes.
   ```python
   from mditonotes import pred_notes

   notes = pred_notes(midi_file)
   print(notes)
   ```

## Dataset

The instrument classification model was trained on an audio dataset containing samples of guitar, drums, piano, and violin. The dataset includes labeled audio files which are used to train and evaluate the model's performance.
Dataset Details :
Drums 700 files
Guitar 1081 files
Piano 1397 files
Violin 700 files

12.9 gb and 3900 files

## Results

The project successfully separates audio tracks into individual components, classifies the musical instruments, converts sound to MIDI, and extracts musical notes. Detailed results, including accuracy metrics and sample outputs, can be found in terminal.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your changes. Ensure that your code adheres to the project's coding standards and includes appropriate tests.



## Acknowledgements

- Thanks to the developers of HDemucs, Librosa, TensorFlow, and other libraries used in this project.
- Special thanks to my project supervisor and peers for their support and guidance.

---

import mido

def extract_notes(file_path):
    notes = []
    mid = mido.MidiFile(file_path)
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on':
                notes.append((msg.note))
    return notes



def midi_to_note(midi_note):
    musical_notes = []
    for note in midi_note:
        # print(note)
        western_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        octave = note // 12
        note_index = note % 12
        western_note = f"{western_notes[note_index]}{octave}"
        musical_notes.append(western_note)
    return musical_notes

def pred_notes(midifile):
    midi_note=extract_notes(midifile)
    musical_notes =midi_to_note(midi_note);
    print("Present :",musical_notes)
    return musical_notes

# pred_notes('seperatedfiles/WhatsApp Audio 2024-03-31 at 5.32.56 PM.mpeg.mid')
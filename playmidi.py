import pygame
import time

def play(midi):
    # Initialize Pygame
    pygame.init()
    print(midi)

    # Load the MIDI file
    pygame.mixer.music.load(midi)

    # Play the MIDI file
    pygame.mixer.music.play()

    # Wait for the MIDI file to finish playing
    while pygame.mixer.music.get_busy():
        time.sleep(1)

    # Clean up
    pygame.mixer.music.stop()
    pygame.quit()

    return "done"

# play("D:/Final/Musical_Notes/separatedfiles/other.mid")
import mido as md
from music21 import *

class MidiToSong():

    def __init__(self):
        pass

    def notes_list(self, tune):
        notes = []
        offset = 0
        for current_note in tune:
            new_note = note.Note(current_note)
            new_note.storedInstrument = instrument.Piano()
            notes.append(new_note)
            offset += 0.5
        return notes

    def save_melody(self, tune, file_name):
        notes = self.notes_list(tune)
        midi_stream = stream.Stream(notes)
        return midi_stream.write('midi',f'{file_name}.mid')

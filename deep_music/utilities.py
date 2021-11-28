import music21
import os
import mido
import pandas as pd
THRESHOLD = 30
from IPython.display import clear_output


def get_properties(filepath):
    file_dict = {}
    songs = os.listdir(filepath)
    counter = 0
    for index, song in enumerate(songs):
        if song.endswith(".mid"):
            try:
                s = filepath
                mid = mido.MidiFile(s+song)
                if mid.length > THRESHOLD:
                        song_key, song_mode = key_finder(s + song)
                        song_bpm = bpm_finder(mid)
                        file_dict[song] = [mid.length, song_key, song_mode, song_bpm]
            except:
                print(f"{song} failed to load")

            clear_output(wait=True)
            print(f"Currently processing {index}/{len(songs)}")

    return file_dict

def key_finder(filename):
    '''function that returns dictionnary tuple of key name and mode
    of midi songs.
    If fails return none'''
    score = music21.converter.parse(filename)
    key = score.analyze('key')
    return (key.tonic.name, key.mode)

def bpm_finder(mid):
    '''function that returns the bpm as integer of midi song.
    If fails return 120 - the default tempo'''

    for track in mid.tracks[0]:
        if (track.dict().get("type") == "set_tempo"):
            return int(mido.tempo2bpm(track.dict().get("tempo")))
    return 120



if __name__ == '__main__':
    print(get_properties)

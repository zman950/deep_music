import music21
import os
import mido
import pandas as pd

def get_properties(filepath):
    file_dict = {}
    songs = os.listdir(filepath)
    counter = 0
    for song in songs:
        s = filepath
        counter +=1
        song_len = length_filter(s+song)
        song_key, song_mode = key_finder(s+song)
        song_bpm = bpm_finder(s+song)
        file_dict[song] = [song_len, song_key, song_mode, song_bpm]
        if counter == 5:
            break
    return file_dict


def length_filter(song, length = 30):
    '''function that removes tracks under a certain length
    '''
    mid = mido.MidiFile(song, clip = True)
    if mid.length > length:
        return mid.length
    return 'None'

def key_finder(song):
    '''function that returns dictionnary tuple of key name and mode
    of midi songs.
    If fails return none'''
    try:
        score = music21.converter.parse(song)
        key = score.analyze('key')
        return (key.tonic.name, key.mode)
    except:
        return 'None'

def bpm_finder(song):
    '''function that returns the bpm as integer of midi song.
    If fails return 120'''
    mid = mido.MidiFile(song, clip = True)
    for msg in mid.tracks[0]:
        counter2 = 0
        if msg.type == 'set_tempo':
            if counter2 == 1:
                break
            counter2 += 1
            tempo = msg.tempo
            bpm = mido.tempo2bpm(tempo)
            return int(bpm)
    return 120


if __name__ == '__main__':
    print(key_finder('/Users/annavaugrante/code/zman950/deep_music/raw_data/Webscrapping/snes'))

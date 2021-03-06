import mido as md
import os
import numpy as np


def training_dataset(note_on, training : list, labels : list):
    #TODO improve how this function works - 20 by default is bad
    """function that returns a training dataset and labels from list_notes"""
    if note_on == None:
        return

    for i in range(20,len(note_on)):
        training.append(note_on[i-20:i])
        labels.append(note_on[i])


def list_notes(midi, n = 50):
    '''generates a list of notes'''
    note = []
    for track in midi:
        if track.dict().get("type") == 'note_on':
            note.append(track.dict().get("note"))

    return note


def generate_dataset(directory, n = 10):
    """returns a train / label dataset from the directory given
    n is the number of train, label values"""
    files = os.listdir(directory)
    X = []
    y = []
    counter = 0
    for file in files:
        if file.endswith(".mid"):
            try:
                counter += 1
                path_to_song = directory + file
                midi_file = md.MidiFile(path_to_song)
                list_of_notes = list_notes(midi_file)
                training_dataset(list_of_notes, X, y)
                print("counter: {}, songs remaining: {}".format(counter, n - counter))
            except:
                print("song failed")
        if counter == n:
            break

    X = reshape_data(X)
    return X, np.array(y)

def reshape_data(X):
    _ = np.array(X)
    _ = _.reshape(( _.shape[0], _.shape[1], 1))
    # print(_)
    return _

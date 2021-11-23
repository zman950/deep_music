import music21
import os

def key_finder(filepath):
    '''function that returns dictionnary tuple of key name and mode
    of midi songs.
    If fails return none'''
    file_dict = {}
    directory = os.listdir('../raw_data/Webscrapping/snes')
    for filepath in directory:
        try:
            p = '../raw_data/Webscrapping/snes/'
            score = music21.converter.parse(p+filepath)
            key = score.analyze('key')
            file_dict[filepath] = (key.tonic.name, key.mode)
        except:
            file_dict[filepath] = 'None'
    return file_dict


if __name__ == '__main__':
    print(key_finder('/Users/annavaugrante/code/zman950/deep_music/raw_data/Webscrapping/snes'))

import numpy as np
import joblib
import tensorflow as tf

model = tf.keras.models.load_model('notebooks/model')


def predict_midi(X, n=20):
    """where X is a sequence of notes
            and n is the number of notes"""
    tune = list(X.reshape(-1, ))
    x = X.reshape(1, n, 1)

    return int(model.predict(x))

def predict_basic(prev_list, prediction, total_list, n_notes):
    '''
        takes in a numpy array of 20 notes, a single note already predicted
        an empty list (could also contain notes), and a number of notes to
        generate
        '''
    if n_notes == 0:
        return total_list

    new_list = []
    for i in prev_list[1:20]:
        new_list.append(int(i))

    new_list = new_list + [prediction]
    prediction_array = np.array(new_list).reshape(20, 1)

    prediction = predict_midi(prediction_array)
    total_list.append(prediction)
    n_notes -= 1

    return predict_basic(prediction_array, prediction, total_list, n_notes)

if __name__ == '__main__':
    X = np.array([[44], [65], [60], [55], [32], [44], [65], [60],
                              [55], [32], [43], [64], [59], [54], [31], [48],
                              [48], [48], [48], [48]])

    prediction = predict_basic(X, 44, [], 20)
    print(prediction)

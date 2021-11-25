import tensorflow as tf
import mido as md
import pandas as pd
import numpy as np
import os
import time
from IPython.display import clear_output
import matplotlib.pyplot as plt
import tensorflow.keras
from sklearn.model_selection import train_test_split
from music21 import *
import joblib
from deep_music.pre_processing_advanced import get_pre_processed_data
from deep_music.models_advanced import simple_rnn, conv1d_model
from termcolor import colored

class Trainer():

    def __init__(self, model, trained = None):
        self.X, self.y = get_pre_processed_data()
        print(self.X.shape)
        print(self.y.shape)
        self.model = model()
        self.replicas()
        self.evaluate(self.X, self.y)
        self.save_model()

    def replicas(self):
        """Defines the run type for the computer """
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
        except ValueError:
            strategy = tf.distribute.get_strategy() # for CPU and single GPU
        return strategy.num_replicas_in_sync

    def evaluate(self, X, y):
        """
        Evaluating function that returns the model -- needs to be tested with model
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
        early_stop = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0)

        eval = self.model.fit(X_train,
                              y_train,
                              epochs=20,
                              batch_size=32 * self.replicas(),
                              validation_data=(X_test, y_test),
                              callbacks=[early_stop])

        return eval

    def load_model(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, prev_list, prediction, total_list, n_notes):
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
        prediction_array = np.array(new_list).reshape(20,1)

        prediction = self.predict_midi(prediction_array)
        total_list.append(prediction)
        n_notes -= 1

        return self.predict(prediction_array, prediction, total_list, n_notes)


    def save_model(self, upload=True, auto_remove=True):
        """Save the model into a .joblib and upload it on Google Storage models folder
        HINTS : use sklearn.joblib (or jbolib) libraries and google-cloud-storage"""

        joblib.dump(self.model, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))


if __name__ == '__main__':
    basic_trainer = Trainer(simple_rnn, "model.joblib")

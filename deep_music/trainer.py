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
from deep_music.pre_processing_basic import generate_dataset
from deep_music.models import simple_rnn, conv1d_model
from termcolor import colored

class Trainer():

    def __init__(self, model, trained = None):
        self.X, self.y = generate_dataset("../raw_data/snes/")
        if trained != None:
            self.load_model(trained)
            prediction = self.predict_midi(self.X)
            print(prediction)
        else:
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

    def predict_midi(self, X):
        starter_notes = X[0]
        print(type(starter_notes))
        tune = list(X[0].reshape(-1, ))
        x = X[0].reshape(1, 20, 1)

        return int(self.model.predict(x))

    def get_20_notes(self, X, prediction):
        new_x = X[1:20].to_list().append(prediction)
        return new_x

    def load_model(self, model_path):
        self.model = joblib.load(model_path)


    def save_model(self, upload=True, auto_remove=True):
        """Save the model into a .joblib and upload it on Google Storage models folder
        HINTS : use sklearn.joblib (or jbolib) libraries and google-cloud-storage"""

        joblib.dump(self.model, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))


if __name__ == '__main__':
    basic_trainer = Trainer(conv1d_model, "model.joblib")

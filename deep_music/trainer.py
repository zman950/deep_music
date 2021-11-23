import tensorflow as tf
import mido as md
import pandas as pd
import numpy as np
import os
import time
from IPython.display import clear_output
import matplotlib.pyplot as plt
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import tensorflow.keras
from tensorflow.keras.layers import SimpleRNN,LSTM,GRU
from sklearn.model_selection import train_test_split
from music21 import *
import joblib
from pre_processing import generate_dataset


class Trainer():

    def __init__(self, model):
        self.model = model
        self.replicas()
        X, y = generate_dataset("../raw_data/snes/")
        self.evaluate(X, y)

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

        model = Trainer.build_model()
        eval = model.fit(X_train, y_train, epochs=20,batch_size=32 * self.strategy.num_replicas_in_sync,
        validation_data=(X_test,y_test),callbacks=[early_stop])

        return eval


    def save_model(self, upload=True, auto_remove=True):
        """Save the model into a .joblib and upload it on Google Storage /models folder
        HINTS : use sklearn.joblib (or jbolib) libraries and google-cloud-storage"""
        joblib.dump(self.model, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))

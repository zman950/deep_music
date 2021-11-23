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

class Trainer():

    def __init__(self):
        pass

    def replicas(self):
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
        except ValueError:
            strategy = tf.distribute.get_strategy() # for CPU and single GPU
        return strategy.num_replicas_in_sync

    def list_notes(self):
        note_on=[]
        n=50
        for m in range(n):
            mid=md.MidiFile(
                '../raw_data/Webscrapping/snes/'+os.listdir(
                '../raw_data/Webscrapping/snes/')[m],clip=True)
            for j in range(len(mid.tracks)):
                for i in mid.tracks[j] :
                    if str(type(i))!="<class 'mido.midifiles.meta.MetaMessage'>" :
                        x=str(i).split(' ')
                        if x[0]=='note_on':
                            note_on.append(int(x[2].split('=')[1]))

            return note_on


    def live_plot_make(x,range_=20,pause_time=0.01,skip_a_do=1):
        for i in range(0,len(x)-range_,skip_a_do):
            plt.figure(figsize=(18,8))
            x_plot=x[i:i+range_]
            y_plot=[i for i in range(range_)]
            fig=plt.plot(y_plot,x_plot,marker='D')
            plt.ylim([min(x),max(x)])
            time.sleep(pause_time)
            clear_output(wait=True)
            plt.show()


    def training_dataset(self):
        training_data=[]
        labels=[]
        for i in range(20,len(self.note_on)):
            training_data.append(self.note_on[i-20:i])
            labels.append(self.note_on[i])
        return training_data, labels


    def build_model(self):
        model=M.Sequential()
        model.add(LSTM(128,input_shape=(10,1)))
        model.add(L.Dense(1,'relu'))
        model.compile(loss='MSE',optimizer='adam')
        return model

    def train(self):
        training_data=np.array(self.training_data)
        training_data=self.training_data.reshape((self.training_data.shape[0],self.training_data.shape[1],1))
        labels=np.array(self.labels)
        return training_data, labels

    def evaluate(self):
        X_train, X_test, y_train, y_test = train_test_split(self.training_data, self.labels, test_size=0.05, random_state=42)
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

import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import tensorflow.keras
from tensorflow.keras.layers import SimpleRNN,LSTM,GRU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def simple_rnn():
    model=M.Sequential()
    model.add(layers.SimpleRNN(units=128, activation='tanh', input_shape=(10,1)))
    model.add(layers.Dense(1, activation="linear"))
    model.compile(loss='mse',optimizer='rmsprop')
    return model

def LSTM_model():
    model=M.Sequential()
    model.add(LSTM(128,input_shape=(10,1)))
    model.add(L.Dense(1,'relu'))
    model.compile(loss='MSE',optimizer='adam')
    return model

def LSTM_softmax_model():
    model=M.Sequential()
    model.add(LSTM(128,input_shape=(10,1),return_sequences=True))
    model.add(LSTM(128))
    model.add(L.Flatten())
    model.add(L.Dense(40,'relu'))
    model.add(L.Dense(40,'softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam')
    return model

def LSTM_2_model():
    model=M.Sequential()
    model.add(LSTM(200,input_shape=(10,1),unroll=True,return_sequences=True))
    model.add(L.Dropout(0.4))
    model.add(LSTM(100))
    model.add(L.Dense(100,'relu'))
    model.add(L.Dropout(0.2))
    model.add(L.Dense(1,'relu'))
    model.compile(loss='MSE',optimizer='adam')
    return model

def gru_model():
    model=M.Sequential()
    model.add(GRU(units=200, input_shape=(10,1), activation='tanh'))
    model.add(L.Dense(1,'relu'))
    model.compile(loss='mse',optimizer='rmsprop')
    return model

def gru_softmax_model():
    model=M.Sequential()
    model.add(GRU(units=128,input_shape=(10,1),return_sequences=True))
    model.add(GRU(128))
    model.add(L.Flatten())
    model.add(L.Dense(40,'relu'))
    model.add(L.Dense(40,'softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop')
    return model

def gru_2_model():
    model=M.Sequential()
    model.add(GRU(units=128,input_shape=(10,1),return_sequences=True))
    model.add(Dropout(0.4))
    model.add(GRU(128, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(TimeDistributedDense(1))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model


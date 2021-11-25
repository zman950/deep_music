import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import tensorflow.keras
from tensorflow.keras.layers import SimpleRNN,LSTM,GRU, Activation
from tensorflow.keras.layers import Conv2D, Conv1D, MaxPooling2D, Flatten, Dense, MaxPooling1D, Conv1DTranspose
from tensorflow.python.keras.layers.core import Reshape

INPUT_SIZE = (60, 96)
def simple_rnn():
    model=M.Sequential()
    model.add(SimpleRNN(units=128, activation='tanh', input_shape=INPUT_SIZE))
    model.add(Dense(50, activation="linear"))
    model.add(Dense(3456, activation="softmax", ))
    model.add(Reshape((36,96)))
    model.compile(loss='mse',optimizer='rmsprop')

    return model

def LSTM_model():
    model=M.Sequential()
    model.add(LSTM(128, input_shape=INPUT_SIZE))
    model.add(L.Dense(1,'relu'))
    model.compile(loss='MSE',optimizer='adam')
    return model

def LSTM_softmax_model():
    model=M.Sequential()
    model.add(LSTM(128,input_shape=INPUT_SIZE,return_sequences=True))
    model.add(LSTM(128))
    model.add(L.Flatten())
    model.add(L.Dense(40,'relu'))
    model.add(L.Dense(40,'softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam')
    return model

def LSTM_2_model():
    model=M.Sequential()
    model.add(LSTM(200,input_shape=INPUT_SIZE,unroll=True,return_sequences=True))
    model.add(L.Dropout(0.4))
    model.add(LSTM(100))
    model.add(L.Dense(100,'relu'))
    model.add(L.Dropout(0.2))
    model.add(L.Dense(1, 'relu'))
    model.compile(loss='MSE', optimizer='adam')
    return model

def gru_model():
    model=M.Sequential()
    model.add(GRU(units=200, input_shape=INPUT_SIZE, activation='tanh'))
    model.add(L.Dense(1,'relu'))
    model.compile(loss='mse',optimizer='rmsprop')
    return model

def gru_softmax_model():
    model=M.Sequential()
    model.add(GRU(units=128,input_shape=INPUT_SIZE,return_sequences=True))
    model.add(GRU(128))
    model.add(L.Flatten())
    model.add(L.Dense(40,'relu'))
    model.add(L.Dense(40,'softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop')
    return model

def gru_2_model():
    model=M.Sequential()
    model.add(GRU(units=128,input_shape=INPUT_SIZE,return_sequences=True))
    model.add(L.Dropout(0.4))
    model.add(GRU(128, return_sequences=True))
    model.add(L.Dropout(0.4))
    model.add(L.TimeDistributedDense(1))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def conv1d_model(time_window_size = 1):
    model = M.Sequential()
    model.add(Conv1D(filters=256, kernel_size=6, padding='same', activation='relu',
                         input_shape=INPUT_SIZE))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(64))
    model.add(Dense(units=time_window_size, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_encoder(latent_dimension):
    '''returns an encoder model, of output_shape equals to latent_dimension'''
    encoder = M.Sequential()
    encoder.add(Conv1D(256, kernel_size=4,input_shape=(10, 1), activation='relu'))
    encoder.add(MaxPooling1D(pool_size=2))
    encoder.add(Conv1D(128, kernel_size=4, activation='relu'))
    encoder.add(MaxPooling1D(pool_size=2))
    encoder.add(Conv1D(32, kerne_size=2, activation='relu'))
    encoder.add(MaxPooling1D(pool_size=2))
    encoder.add(Flatten())
    encoder.add(Dense(latent_dimension, activation='tanh'))
    return encoder

# def build_decoder(latent_dimension):
#     decoder = M.Sequential()
#     decoder.add(Dense(7*7*8, activation='tanh', input_shape=(latent_dimension,)))
#     decoder.add(Reshape((7, 7, 8)))
#     decoder.add(Conv1DTranspose(8, (2, 2), strides=2, padding='same', activation='relu'))
#     decoder.add(Conv1DTranspose(1, (2, 2), strides=2, padding='same', activation='relu'))
#     return decoder

# def build_autoencoder(encoder, decoder):
#     inp = Input((28, 28,1))
#     encoded = encoder(inp)
#     decoded = decoder(encoded)
#     autoencoder = Model(inp, decoded)
#     return autoencoder

# def compile_autoencoder(autoencoder):
#     autoencoder_comp = autoencoder.compile(loss='mse',
#                   optimizer='adam')
#     return autoencoder_comp

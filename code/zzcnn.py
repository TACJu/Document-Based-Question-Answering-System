import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Flatten, Conv1D, MaxPooling1D, Concatenate, Dot
from keras.engine.topology import Layer
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import numpy as np
batch_size = 128
input_length = 200
kernel_size = 5

def build_model(embedding_matrix):
                        
    doc_embedding = Embedding(185674, 300, weights=[embedding_matrix], input_length=input_length, trainable=False)
    doc_input = Input(shape=(input_length,), dtype='int32')
    doc_sequences = doc_embedding(doc_input)
    D = Conv1D(64, kernel_size, activation='relu', padding='same')(doc_sequences)
    D = MaxPooling1D(kernel_size)(D)
    # The more, the better?
    D = Conv1D(64, kernel_size, activation='relu', padding='same')(D)
    D = Conv1D(64, kernel_size, activation='relu', padding='same')(D)
    D = Conv1D(64, kernel_size, activation='relu', padding='same')(D)
    D = AveragePooling1D(kernel_size)(D)
    # The more, the better?
    D = Flatten()(D)

    query_embedding = Embedding(185674, 300, weights=[embedding_matrix], input_length=input_length, trainable=False)
    query_input = Input(shape=(input_length,), dtype='int32')
    query_sequences = query_embedding(query_input)
    Q = Conv1D(64, kernel_size, activation='relu', padding='same')(query_sequences)
    Q = MaxPooling1D(kernel_size)(Q)
    # The more, the better?
    Q = Conv1D(64, kernel_size, activation='relu', padding='same')(Q)
    Q = Conv1D(64, kernel_size, activation='relu', padding='same')(Q)
    Q = Conv1D(64, kernel_size, activation='relu', padding='same')(Q)
    Q = AveragePooling1D(kernel_size)(Q)
    # The more, the better?
    Q = Flatten()(Q)

    left = Dense(2560, use_bias=False)(Q)
    sim = Dot(1)([left, D])

    # new sim, change the operation order
    new_left = Dense(2560, use_bias=False)(D)
    new_sim = Dot(1)([new_left, Q])
    concat = Concatenate()([Q, D, sim, new_sim])
    # new sim, change the operation order


    x = Dense(128, activation='relu')(concat)

    preds = Dense(1, activation='sigmoid')(x)

    model = Model([query_input, doc_input], preds)
    model.summary()
    return model

if __name__ == "__main__":

    X_train_Q = np.load('../data/numpy_array/train_Q_index.npy') # shape (264416, 1000)
    X_train_A = np.load('../data/numpy_array/train_A_index.npy')
    X_train_Q = X_train_Q[:,:200]
    X_train_A = X_train_A[:,:200]
    
    #X_train = np.concatenate((X_train_Q, X_train_A), axis=1)
    X_val_Q = np.load('../data/numpy_array/validation_Q_index.npy')
    X_val_A = np.load('../data/numpy_array/validation_A_index.npy')
    X_val_Q = X_val_Q[:,:200]
    X_val_A = X_val_A[:,:200]
    #X_val = np.concatenate((X_val_Q, X_val_A), axis=1)
    Y_train = np.load('../data/numpy_array/train_label.npy')
    Y_val = np.load('../data/numpy_array/validation_label.npy')
    embedding_matrix = np.load('../data/numpy_array/word_vector.npy')
    
    cw = {0:1, 1:20}
    filepath='../model/net_model/model_{epoch:02d}-{val_acc:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    model = build_model(embedding_matrix)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit([X_train_Q, X_train_A], Y_train, validation_data=([X_val_Q, X_val_A], Y_val), callbacks=[checkpoint], epochs=10, batch_size=batch_size, class_weight = cw)
    
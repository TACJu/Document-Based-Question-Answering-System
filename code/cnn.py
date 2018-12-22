import numpy as np
import os
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Flatten, Conv1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def build_model(embedding_matrix):
    
    embedding_layer = Embedding(185674, 300, weights=[embedding_matrix], input_length=400, trainable=False)
    sequence_input = Input(shape=(400,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(1, activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    plot_model(model, to_file='./net_structure/cnn.png')
    return model

if __name__ == "__main__":

    X_train_Q = np.load('../data/numpy_array/train_Q_index.npy')
    X_train_A = np.load('../data/numpy_array/train_A_index.npy')
    X_train_Q = X_train_Q[:,:200]
    X_train_A = X_train_A[:,:200]
    print(X_train_Q.shape)
    X_train = np.concatenate((X_train_Q, X_train_A), axis=1)
    X_val_Q = np.load('../data/numpy_array/validation_Q_index.npy')
    X_val_A = np.load('../data/numpy_array/validation_A_index.npy')
    X_val_Q = X_val_Q[:,:200]
    X_val_A = X_val_A[:,:200]
    X_val = np.concatenate((X_val_Q, X_val_A), axis=1)
    Y_train = np.load('../data/numpy_array/train_label.npy')
    Y_val = np.load('../data/numpy_array/validation_label.npy')
    embedding_matrix = np.load('../data/numpy_array/word_vector.npy')
    
    cw = {0:1, 1:20}
    filepath='../model/net_model/model_{epoch:02d}-{val_acc:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    model = build_model(embedding_matrix)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), callbacks=[checkpoint], epochs=3, batch_size=100, class_weight=cw)
        
    
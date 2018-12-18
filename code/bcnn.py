import numpy as np
import os
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Flatten, Conv1D, AveragePooling1D, Concatenate, Dot
from keras.callbacks import ModelCheckpoint
import numpy as np

batch_size = 128
input_length = 200
kernel_size = 5

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def build_model(embedding_matrix):
    left_embedding = Embedding(185674, 300, weights=[embedding_matrix], input_length=input_length, trainable=False)
    left_input = Input(shape=(input_length,), dtype='int32')
    left_sequences = left_embedding(left_input)
    left = Conv1D(64, kernel_size, activation='relu', padding='causal')(left_sequences)
    left = AveragePooling1D(kernel_size)(left)
    left = Conv1D(64, kernel_size, activation='relu', padding='causal')(left)
    left = AveragePooling1D(40)(left)
    left = Flatten()(left)

    right_embedding = Embedding(185674, 300, weights=[embedding_matrix], input_length=input_length, trainable=False)
    right_input = Input(shape=(input_length,), dtype='int32')
    right_sequences = right_embedding(right_input)
    right = Conv1D(64, kernel_size, activation='relu', padding='causal')(right_sequences)
    right = AveragePooling1D(kernel_size)(right)
    right = Conv1D(64, kernel_size, activation='relu', padding='causal')(right)
    right = AveragePooling1D(40)(right)
    right = Flatten()(right)

    #  note that abcnn use all-ap average pooling in the last, which should be followed by a logistic regression,
    #  this process could be change to some more efficient way
    sim = Dot(1)([left, right])  # magical modified
    concat = Concatenate()([left, right, sim])  # magical modified

    x = Dense(128, activation='relu')(concat)
    x = Dense(128, activation='relu')(x)

    preds = Dense(1, activation='sigmoid')(x)

    model = Model([left_input, right_input], preds)
    return model


if __name__ == "__main__":
    X_train_Q = np.load('../data/numpy_array/train_Q_index.npy')  # shape (264416, 1000)
    X_train_A = np.load('../data/numpy_array/train_A_index.npy')
    X_train_Q = X_train_Q[:, :200]
    X_train_A = X_train_A[:, :200]

    X_val_Q = np.load('../data/numpy_array/validation_Q_index.npy')
    X_val_A = np.load('../data/numpy_array/validation_A_index.npy')
    X_val_Q = X_val_Q[:, :200]
    X_val_A = X_val_A[:, :200]
    Y_train = np.load('../data/numpy_array/train_label.npy')
    Y_val = np.load('../data/numpy_array/validation_label.npy')
    embedding_matrix = np.load('../data/numpy_array/word_vector.npy')

    cw = {0: 1, 1: 20}
    filepath = '../model/net_model/model_{epoch:02d}-{val_acc:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False,
                                 mode='auto', period=1)
    model = build_model(embedding_matrix)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit([X_train_Q, X_train_A], Y_train, validation_data=([X_val_Q, X_val_A], Y_val), callbacks=[checkpoint],
              epochs=10, batch_size=batch_size, class_weight=cw)

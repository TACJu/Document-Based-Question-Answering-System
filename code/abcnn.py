import os
import tensorflow
from tensorflow import keras
from keras import initializers,regularizers,constraints
from keras import backend as K
from keras.models import Sequential, Model
from keras.engine.topology import Layer
from keras.layers import Input, Embedding, Dense, Flatten, Conv1D, AveragePooling1D, Concatenate, Dot, Permute
from keras.callbacks import ModelCheckpoint
import numpy as np

batch_size = 128
input_length = 200
kernel_size = 5

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def dot_product(x, y):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(y)), axis=-1)
    else:
        return K.dot(x, y)


class Attention_matrix(Layer):
    def build(self, input_shape):
        super(Attention_matrix, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        left_phrase = x[0]
        right_phrase = x[1]
        return 1. / K.maximum(1. + K.sqrt(
            - 2 * K.batch_dot(left_phrase, right_phrase, axes=[2, 2]) +
            K.expand_dims(K.sum(K.square(left_phrase), axis=2), 2) +
            K.expand_dims(K.sum(K.square(right_phrase), axis=2), 1)
        ), K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[0][1]


#def l2_match_score(input_pair):
#    left_phrase, right_phrase = input_pair
#    return 1. / K.maximum(1. + K.sqrt(
#        - 2 * K.batch_dot(left_phrase, right_phrase, axes=[2, 2]) +
#        K.expand_dims(K.sum(K.square(left_phrase), axis=2), 2) +
#        K.expand_dims(K.sum(K.square(right_phrase), axis=2), 1)
#    ), K.epsilon())


class Attention_weight(Layer):
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1, input_shape[2], 300),
                                      initializer='uniform',
                                      trainable=True)
        super(Attention_weight, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        batch_kernel = K.repeat_elements(self.kernel, batch_size, axis=0)
        output = K.batch_dot(x, batch_kernel, axes=[2, 1])
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], 300

#class Trans(Layer):
#    def build(self, input_shape):
#        super(Trans, self).build(input_shape)  # Be sure to call this somewhere!
#    def call(self, x):
#        output = K.transpose(x)
#        return output
#    def compute_output_shape(self, input_shape):
#        return input_shape[0], input_shape[2], input_shape[1]

def build_model(embedding_matrix):
    left_embedding = Embedding(185674, 300, weights=[embedding_matrix], input_length=input_length, trainable=False)
    left_input = Input(shape=(input_length,), dtype='int32')
    left_sequences = left_embedding(left_input)

    right_embedding = Embedding(185674, 300, weights=[embedding_matrix], input_length=input_length, trainable=False)
    right_input = Input(shape=(input_length,), dtype='int32')
    right_sequences = right_embedding(right_input)

    # abcnn1
    A1 = Attention_matrix()([left_sequences, right_sequences])
    A1_t = Permute((2, 1))(A1)

    left_attention = Attention_weight()(A1_t)
    left = Concatenate(axis=-1)([left_attention, left_sequences])
    left = Conv1D(64, kernel_size, activation='relu', padding='same')(left)

    right_attention = Attention_weight()(A1)
    right = Concatenate(axis=-1)([right_attention, right_sequences])
    right = Conv1D(64, kernel_size, activation='relu', padding='same')(right)
    # abcnn2
    #A2 = match_score_merge()([left, right])

    left = AveragePooling1D(kernel_size, strides=1)(left)
    left = Conv1D(64, kernel_size, activation='relu', padding='same')(left)
    left = AveragePooling1D(196)(left)
    left = Flatten()(left)

    right = AveragePooling1D(kernel_size, strides=1)(right)
    right = Conv1D(64, kernel_size, activation='relu', padding='same')(right)
    right = AveragePooling1D(196)(right)
    right = Flatten()(right)

    #  note that abcnn use all-ap average pooling in the last, which should be followed by a logistic regression,
    #  this process could be change to some more efficient way
    # sim = Dot(1)([left, right])  # magical modified
    concat = Concatenate()([left, right])  # magical modified

    x = Dense(256, activation='relu')(concat)
    x = Dense(128, activation='relu')(x)

    preds = Dense(1, activation='sigmoid')(x)

    model = Model([left_input, right_input], preds)
    return model


if __name__ == "__main__":
    X_train_Q = np.load('../data/numpy_array/train_Q_index.npy')  # shape (264416, 1000)
    X_train_A = np.load('../data/numpy_array/train_A_index.npy')
    X_train_Q = X_train_Q[96:, :200]
    X_train_A = X_train_A[96:, :200]

    X_val_Q = np.load('../data/numpy_array/validation_Q_index.npy')
    X_val_A = np.load('../data/numpy_array/validation_A_index.npy')
    X_val_Q = X_val_Q[61:, :200]
    X_val_A = X_val_A[61:, :200]
    Y_train = np.load('../data/numpy_array/train_label.npy')
    Y_val = np.load('../data/numpy_array/validation_label.npy')
    Y_train = Y_train[96:]
    Y_val = Y_val[61:]
    embedding_matrix = np.load('../data/numpy_array/word_vector.npy')

    cw = {0: 1, 1: 20}
    filepath = '../model/abc_model/model_{epoch:02d}-{val_acc:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False,
                                 mode='auto', period=1)
    model = build_model(embedding_matrix)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit([X_train_Q, X_train_A], Y_train, validation_data=([X_val_Q, X_val_A], Y_val), callbacks=[checkpoint],
              epochs=20, batch_size=batch_size, class_weight=cw)
import numpy as np
import os

import keras
from keras import backend as K
from keras import initializers,regularizers,constraints
from keras.models import Model, Sequential
from keras.engine.topology import Layer
from keras.layers import Dense, Input, Embedding, LSTM, GRU, Bidirectional, TimeDistributed, Concatenate, BatchNormalization, Lambda
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model
from keras.utils import CustomObjectScope

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(Attention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

def build_model(embedding_matrix):
    
    embedding_layer = Embedding(185674, 300, weights=[embedding_matrix], input_length=200, trainable=False, mask_zero=True)

    input_query = Input(shape=(200,), dtype='int32')
    embedded_query = embedding_layer(input_query)
    lstm_query = Bidirectional(LSTM(32, return_sequences=True))(embedded_query)
    #lstm_Q = Bidirectional(LSTM(32))(lstm_query)
    #dense_query = TimeDistributed(Dense(200))(lstm_query)
    attn_query = Attention()(lstm_query)

    input_answer = Input(shape=(200,), dtype='int32')
    embedding_answer = embedding_layer(input_answer)
    lstm_answer = Bidirectional(LSTM(32, return_sequences=True))(embedding_answer)
    #lstm_A = Bidirectional(LSTM(32))(lstm_answer)
    #dense_answer = TimeDistributed(Dense(200))(lstm_answer)
    attn_answer = Attention()(lstm_answer)

    concat = Concatenate()([attn_query, attn_answer])
    x = Lambda(lambda y:1-y)(concat)
    x = Dense(128, activation='relu')(x)
    pred = Dense(1, activation='sigmoid')(x)
    model = Model([input_query, input_answer], pred)

    model.summary()
    return model

if __name__ == "__main__":

    X_train_Q = np.load('../data/numpy_array/train_Q_index.npy')
    X_train_A = np.load('../data/numpy_array/train_A_index.npy')
    X_train_Q = X_train_Q[:,:200]
    X_train_A = X_train_A[:,:200]
    #X_train = np.concatenate((X_train_Q, X_train_A), axis=1)
    X_val_Q = np.load('../data/numpy_array/validation_Q_index.npy')
    X_val_A = np.load('../data/numpy_array/validation_A_index.npy')
    X_val_Q = X_val_Q[:,:200]
    X_val_A = X_val_A[:,:200]
    #X_val = np.concatenate((X_val_Q, X_val_A), axis=1)
    #X_train = np.expand_dims(X_train, axis=1)
    #X_val = np.expand_dims(X_val, axis=1)
    Y_train = np.load('../data/numpy_array/train_label.npy')
    Y_val = np.load('../data/numpy_array/validation_label.npy')
    embedding_matrix = np.load('../data/numpy_array/word_vector.npy')

    cw = {0:1, 1:20}
    filepath='../model/net_model_tmp/model_{epoch:02d}-{val_acc:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    #model = build_model(embedding_matrix)
    with CustomObjectScope({'Attention': Attention()}):
        model = load_model('../model/net_model/model_10-0.84.hdf5')
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit([X_train_Q, X_train_A], Y_train, validation_data=([X_val_Q, X_val_A], Y_val), callbacks=[checkpoint], epochs=10, batch_size=128, class_weight=cw)
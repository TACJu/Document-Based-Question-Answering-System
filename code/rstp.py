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
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# class MergeLayer(Layer):
#     def build(self, input_shape):
#         assert isinstance(input_shape, list)
#         # Create a trainable weight variable for this layer.
#         self.kernel = self.add_weight(name='kernel', 
#                                       shape=((input_shape[0][1], input_shape[0][1])),
#                                       initializer='uniform',
#                                       trainable=True)
#         self.dim = input_shape[0][1]
#         super(MergeLayer, self).build(input_shape)  # Be sure to call this somewhere!

#     def call(self, x):
#         a, b = x
#         br = K.reshape(b, (-1, self.dim, 1))
        
#         sim = np.zeros((batch_size, 1))
#         for i in range(batch_size):
#              ar = K.reshape(a[i], (-1, self.dim))
#              t = K.dot(K.dot(ar, self.kernel), br[i])
#              sim[i][0] = K.eval(t)[0][0]
#         sim = K.variable(value=sim)
#         transform_left = K.dot(a, self.kernel)
#         sim = tf.reduce_sum()
#         return K.concatenate([a, sim, b])

#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], 1)

def build_model(embedding_matrix):
                        
    doc_embedding = Embedding(74925, 300, weights=[embedding_matrix], input_length=input_length, trainable=False)
    doc_input = Input(shape=(input_length,), dtype='int32')
    doc_sequences = doc_embedding(doc_input)
    D = Conv1D(64, 5, activation='relu', padding='same')(doc_sequences)
    D = MaxPooling1D(5)(D)
    D = Flatten()(D)

    query_embedding = Embedding(74925, 300, weights=[embedding_matrix], input_length=input_length, trainable=False)
    query_input = Input(shape=(input_length,), dtype='int32')
    query_sequences = query_embedding(query_input)
    Q = Conv1D(64, 5, activation='relu', padding='same')(query_sequences)
    Q = MaxPooling1D(5)(Q)
    Q = Flatten()(Q)

    left = Dense(2560, use_bias=False)(Q)
    sim = Dot(1)([left, D])
    concat = Concatenate()([Q, D, sim])

    x = Dense(128, activation='relu')(concat)

    preds = Dense(1, activation='sigmoid')(x)

    model = Model([query_input, doc_input], preds)
    model.summary()
    return model

if __name__ == "__main__":

    X_train_Q = np.load('../data/numpy_array/new_train_Q_index.npy') # shape (264416, 1000)
    X_train_A = np.load('../data/numpy_array/new_train_A_index.npy')
    #X_train_Q = X_train_Q[:,:200]
    #X_train_A = X_train_A[:,:200]
    
    #X_train = np.concatenate((X_train_Q, X_train_A), axis=1)
    X_val_Q = np.load('../data/numpy_array/new_validation_Q_index.npy')
    X_val_A = np.load('../data/numpy_array/new_validation_A_index.npy')
    #X_val_Q = X_val_Q[:,:200]
    #X_val_A = X_val_A[:,:200]
    #X_val = np.concatenate((X_val_Q, X_val_A), axis=1)
    Y_train = np.load('../data/numpy_array/train_label.npy')
    Y_val = np.load('../data/numpy_array/validation_label.npy')
    embedding_matrix = np.load('../data/numpy_array/new_word_vector.npy')
    
    cw = {0:1, 1:20}
    filepath='../model/net_model/model_{epoch:02d}-{val_acc:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    model = build_model(embedding_matrix)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit([X_train_Q, X_train_A], Y_train, validation_data=([X_val_Q, X_val_A], Y_val), callbacks=[checkpoint], epochs=10, batch_size=batch_size, class_weight = cw)
    
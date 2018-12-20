import os
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Convolution1D, Convolution2D, AveragePooling1D, GlobalAveragePooling1D, Dense, \
    Lambda, TimeDistributed, RepeatVector, Permute, ZeroPadding1D, ZeroPadding2D, Reshape, Dropout, \
    BatchNormalization, Embedding, Concatenate, Multiply
from keras.engine.topology import Layer
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import numpy as np

batch_size = 128

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


def ABCNN(
        embedding_matrix, origin_dimensions, left_seq_len, right_seq_len, embed_dimensions, nb_filter, filter_widths,
        depth=2, dropout=0.4, abcnn_1=True, abcnn_2=True, collect_sentence_representations=False, batch_normalize=True):
    assert depth >= 1, "Need at least one layer to build ABCNN"
    assert not (depth == 1 and abcnn_2), "Cannot build ABCNN-2 with only one layer!"
    if type(filter_widths) == int:
        filter_widths = [filter_widths] * depth
    assert len(filter_widths) == depth

    left_sentence_representations =  []
    right_sentence_representations = []

    left_embedding = Embedding(origin_dimensions, embed_dimensions, weights=[embedding_matrix], input_length=left_seq_len, trainable=False)
    right_embedding = Embedding(origin_dimensions, embed_dimensions, weights=[embedding_matrix], input_length=right_seq_len, trainable=False)
    
    left_input = Input(shape=(left_seq_len, ))
    right_input = Input(shape=(right_seq_len, ))

    left_embed = left_embedding(left_input)
    right_embed = right_embedding(right_input)

    filter_width = filter_widths[0]
    if abcnn_1:
        A1 = Attention_matrix()([left_embed, right_embed])
        A1_t = Permute((2, 1))(A1)

        left_attention = Attention_weight()(A1_t)
        left = Concatenate(axis=-1)([left_attention, left_embed])
        left = Convolution1D(64, filter_width, activation='relu', padding='same')(left)

        right_attention = Attention_weight()(A1)
        right = Concatenate(axis=-1)([right_attention, right_embed])
        right = Convolution1D(64, filter_width, activation='relu', padding='same')(right)

    conv_left = Dropout(dropout)(left)
    conv_right = Dropout(dropout)(right)

    pool_left = AveragePooling1D(pool_size=filter_width, strides=1, padding="same")(conv_left)
    pool_right = AveragePooling1D(pool_size=filter_width, strides=1, padding="same")(conv_right)

    assert pool_left._keras_shape[1] == left_seq_len, "%s != %s" % (pool_left._keras_shape[1], left_seq_len)
    assert pool_right._keras_shape[1] == right_seq_len, "%s != %s" % (pool_right._keras_shape[1], right_seq_len)

    if collect_sentence_representations or depth == 1:  # always collect last layers global representation
        left_sentence_representations.append(GlobalAveragePooling1D()(conv_left))
        right_sentence_representations.append(GlobalAveragePooling1D()(conv_right))

    filter_width = filter_widths[1]
    pool_left = ZeroPadding1D(filter_width - 1)(pool_left)
    pool_right = ZeroPadding1D(filter_width - 1)(pool_right)
    # Wide convolution
    conv_left = Convolution1D(nb_filter, filter_width, activation="relu", padding="valid")(pool_left)
    conv_right = Convolution1D(nb_filter, filter_width, activation="relu", padding="valid")(pool_right)

    if abcnn_2:
        conv_match_score = Attention_matrix()([conv_left, conv_right])

        # compute attention
        conv_attention_left = Lambda(lambda match: K.sum(match, axis=-1), output_shape=(conv_match_score._keras_shape[1],))(conv_match_score)
        conv_attention_right = Lambda(lambda match: K.sum(match, axis=-2), output_shape=(conv_match_score._keras_shape[2],))(conv_match_score)

        conv_attention_left = Permute((2, 1))(RepeatVector(nb_filter)(conv_attention_left))
        conv_attention_right = Permute((2, 1))(RepeatVector(nb_filter)(conv_attention_right))

        # apply attention multiply each value by the sum of it's respective attention row/column
        conv_left = Multiply()([conv_left, conv_attention_left])
        conv_right = Multiply()([conv_right, conv_attention_right])

    conv_left = Dropout(dropout)(conv_left)
    conv_right = Dropout(dropout)(conv_right)

    pool_left = AveragePooling1D(pool_size=filter_width, strides=1, padding="valid")(conv_left)
    pool_right = AveragePooling1D(pool_size=filter_width, strides=1, padding="valid")(conv_right)

    assert pool_left._keras_shape[1] == left_seq_len
    assert pool_right._keras_shape[1] == right_seq_len

    left_sentence_representations.append(GlobalAveragePooling1D()(conv_left))
    right_sentence_representations.append(GlobalAveragePooling1D()(conv_right))

    # ###################### #
    # ### END OF ABCNN-2 ### #
    # ###################### #

    # Merge collected sentence representations if necessary
    left_sentence_rep = left_sentence_representations.pop(-1)
    if left_sentence_representations:
        left_sentence_rep = Concatenate()([left_sentence_rep] + left_sentence_representations)

    right_sentence_rep = right_sentence_representations.pop(-1)
    if right_sentence_representations:
        right_sentence_rep = Concatenate()([right_sentence_rep] + right_sentence_representations)

    global_representation = Concatenate()([left_sentence_rep, right_sentence_rep])
    global_representation = Dropout(dropout)(global_representation)

    # Add logistic regression on top.
    classify = Dense(1, activation="sigmoid")(global_representation)
    model = Model([left_input, right_input], outputs=classify)
    # model.summary()
    return model

if __name__ == "__main__":
    query_seq_len = 300
    doc_seq_len = 300
    X_train_Q = np.load('../data/numpy_array/train_Q_index.npy')  # shape (264416, 1000)
    X_train_A = np.load('../data/numpy_array/train_A_index.npy')
    X_train_Q = X_train_Q[96:, :query_seq_len]
    X_train_A = X_train_A[96:, :doc_seq_len]

    X_val_Q = np.load('../data/numpy_array/validation_Q_index.npy')
    X_val_A = np.load('../data/numpy_array/validation_A_index.npy')
    X_val_Q = X_val_Q[61:, :query_seq_len]
    X_val_A = X_val_A[61:, :doc_seq_len]
    Y_train = np.load('../data/numpy_array/train_label.npy')
    Y_val = np.load('../data/numpy_array/validation_label.npy')
    Y_train = Y_train[96:]
    Y_val = Y_val[61:]
    embedding_matrix = np.load('../data/numpy_array/word_vector.npy')

    cw = {0: 1, 1: 20}
    filepath = '../model/short_net_model/model_{epoch:02d}-{val_acc:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False,
                                 mode='auto', period=1)

    embed_dimensions = 300
    origin_dimensions = 185674
    nb_filter = 300
    filter_width = [3, 4]
    
    model = ABCNN(
        embedding_matrix=embedding_matrix, origin_dimensions=origin_dimensions, left_seq_len=query_seq_len, right_seq_len=doc_seq_len,
        embed_dimensions=embed_dimensions, nb_filter=nb_filter, filter_widths=filter_width,
        collect_sentence_representations=True, abcnn_1=True, abcnn_2=True
    )
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit([X_train_Q, X_train_A], Y_train, validation_data=([X_val_Q, X_val_A], Y_val), callbacks=[checkpoint],
              epochs=10, batch_size=batch_size, class_weight=cw)

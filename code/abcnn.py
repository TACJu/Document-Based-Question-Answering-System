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

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def dot_product(x, y):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(y)), axis=-1)
    else:
        return K.dot(x, y)

################################################ Keras Merge ##########################################################
#######################################################################################################################

class keras_merge(Layer):
    def __init__(self, **kwargs):
        super(keras_merge, self).__init__(**kwargs)
        self.supports_masking = True

    def _merge_function(self, inputs):
        raise NotImplementedError

    def _compute_elemwise_op_output_shape(self, shape1, shape2):
        if None in [shape1, shape2]:
            return None
        elif len(shape1) < len(shape2):
            return self._compute_elemwise_op_output_shape(shape2, shape1)
        elif not shape2:
            return shape1
        output_shape = list(shape1[:-len(shape2)])
        for i, j in zip(shape1[-len(shape2):], shape2):
            if i is None or j is None:
                output_shape.append(None)
            elif i == 1:
                output_shape.append(j)
            elif j == 1:
                output_shape.append(i)
            else:
                if i != j:
                    raise ValueError('Operands could not be broadcast '
                                     'together with shapes ' +
                                     str(shape1) + ' ' + str(shape2))
                output_shape.append(i)
        return tuple(output_shape)

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list):
            raise ValueError('A merge layer should be called '
                             'on a list of inputs.')
        if len(input_shape) < 2:
            raise ValueError('A merge layer should be called '
                             'on a list of at least 2 inputs. '
                             'Got ' + str(len(input_shape)) + ' inputs.')
        batch_sizes = [s[0] for s in input_shape if s is not None]
        batch_sizes = set(batch_sizes)
        batch_sizes -= set([None])
        if len(batch_sizes) > 1:
            raise ValueError('Can not merge tensors with different '
                             'batch sizes. Got tensors with shapes : ' +
                             str(input_shape))
        if input_shape[0] is None:
            output_shape = None
        else:
            output_shape = input_shape[0][1:]
        for i in range(1, len(input_shape)):
            if input_shape[i] is None:
                shape = None
            else:
                shape = input_shape[i][1:]
            output_shape = self._compute_elemwise_op_output_shape(output_shape,
                                                                  shape)
        # If the inputs have different ranks, we have to reshape them
        # to make them broadcastable.
        if None not in input_shape and len(set(map(len, input_shape))) == 1:
            self._reshape_required = False
        else:
            self._reshape_required = True

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('A merge layer should be called '
                             'on a list of inputs.')
        if self._reshape_required:
            reshaped_inputs = []
            input_ndims = list(map(K.ndim, inputs))
            if None not in input_ndims:
                # If ranks of all inputs are available,
                # we simply expand each of them at axis=1
                # until all of them have the same rank.
                max_ndim = max(input_ndims)
                for x in inputs:
                    x_ndim = K.ndim(x)
                    for _ in range(max_ndim - x_ndim):
                        x = K.expand_dims(x, 1)
                    reshaped_inputs.append(x)
                return self._merge_function(reshaped_inputs)
            else:
                # Transpose all inputs so that batch size is the last dimension.
                # (batch_size, dim1, dim2, ... ) -> (dim1, dim2, ... , batch_size)
                transposed = False
                for x in inputs:
                    x_ndim = K.ndim(x)
                    if x_ndim is None:
                        x_shape = K.shape(x)
                        batch_size = x_shape[0]
                        new_shape = K.concatenate([x_shape[1:],
                                                   K.expand_dims(batch_size)])
                        x_transposed = K.reshape(x, K.stack([batch_size,
                                                             K.prod(x_shape[1:])]))
                        x_transposed = K.permute_dimensions(x_transposed, (1, 0))
                        x_transposed = K.reshape(x_transposed, new_shape)
                        reshaped_inputs.append(x_transposed)
                        transposed = True
                    elif x_ndim > 1:
                        dims = list(range(1, x_ndim)) + [0]
                        reshaped_inputs.append(K.permute_dimensions(x, dims))
                        transposed = True
                    else:
                        # We don't transpose inputs if they are
                        # 1D vectors or scalars.
                        reshaped_inputs.append(x)
                y = self._merge_function(reshaped_inputs)
                y_ndim = K.ndim(y)
                if transposed:
                    # If inputs have been transposed,
                    # we have to transpose the output too.
                    if y_ndim is None:
                        y_shape = K.shape(y)
                        y_ndim = K.shape(y_shape)[0]
                        batch_size = y_shape[y_ndim - 1]
                        new_shape = K.concatenate([K.expand_dims(batch_size),
                                                   y_shape[:y_ndim - 1]])
                        y = K.reshape(y, (-1, batch_size))
                        y = K.permute_dimensions(y, (1, 0))
                        y = K.reshape(y, new_shape)
                    elif y_ndim > 1:
                        dims = [y_ndim - 1] + list(range(y_ndim - 1))
                        y = K.permute_dimensions(y, dims)
                return y
        else:
            return self._merge_function(inputs)

    def compute_output_shape(self, input_shape):
        if input_shape[0] is None:
            output_shape = None
        else:
            output_shape = input_shape[0][1:]
        for i in range(1, len(input_shape)):
            if input_shape[i] is None:
                shape = None
            else:
                shape = input_shape[i][1:]
            output_shape = self._compute_elemwise_op_output_shape(output_shape,
                                                                  shape)
        batch_sizes = [s[0] for s in input_shape if s is not None]
        batch_sizes = set(batch_sizes)
        batch_sizes -= set([None])
        if len(batch_sizes) == 1:
            output_shape = (list(batch_sizes)[0],) + output_shape
        else:
            output_shape = (None,) + output_shape
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        if not isinstance(mask, list):
            raise ValueError('`mask` should be a list.')
        if not isinstance(inputs, list):
            raise ValueError('`inputs` should be a list.')
        if len(mask) != len(inputs):
            raise ValueError('The lists `inputs` and `mask` '
                             'should have the same length.')
        if all([m is None for m in mask]):
            return None
        masks = [K.expand_dims(m, 0) for m in mask if m is not None]
        return K.all(K.concatenate(masks, axis=0), axis=0, keepdims=False)

################################################ Keras Merge ##########################################################
#######################################################################################################################

class match_score_merge(keras_merge):
    def _merge_function(self, inputs):
        left_phrase = inputs[0]
        right_phrase = inputs[1]
        return 1. / K.maximum(1. + K.sqrt(
            - 2 * K.batch_dot(left_phrase, right_phrase, axes=[2, 2]) +
            K.expand_dims(K.sum(K.square(left_phrase), axis=2), 2) +
            K.expand_dims(K.sum(K.square(right_phrase), axis=2), 1)
        ), K.epsilon())

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
        output = K.batch_dot(x, self.kernel, axes=[1,2])
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
    A1 = match_score_merge()([left_sequences, right_sequences])
    A1_t = Permute((2, 1))(A1)
    left_attention = Attention_weight()(A1_t)
    left = Concatenate(axis=-1)(left_attention, left_sequences)
    left = Conv1D(64, kernel_size, activation='relu', padding='same')(left)

    right_attention = Attention_weight()(A1)
    right = Concatenate(axis=-1)(right_attention, right_sequences)
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

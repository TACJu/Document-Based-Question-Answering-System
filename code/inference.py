import numpy as np
import os
import keras
from keras import backend as K
from keras import initializers,regularizers,constraints
from keras.models import Model
from keras.engine.topology import Layer
from keras.layers import Dense, Input, Embedding, GRU, Bidirectional, TimeDistributed
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

def MRR(pred, label):
    global num
    Q = len(num)
    mrr = 0
    index = 0
    for i in num:
        tmp_pred = pred[index:index + i]
        tmp_label = label[index:index + i]
        true_index = np.argmax(tmp_label)
        tmp_rank = np.argsort(-tmp_pred)
        rank = np.argwhere(tmp_rank == true_index)[0][0] + 1
        mrr += 1/rank
        index += i
    mrr /= Q
    return mrr

X_val_Q = np.load('../data/numpy_array/validation_Q_index.npy')
X_val_A = np.load('../data/numpy_array/validation_A_index.npy')
num = []

length = len(X_val_Q)
print(length)
count = 1
for i in range(length):
    if i == 0:
        continue
    else:
        if (X_val_Q[i - 1] == X_val_Q[i]).all() == True:
            count += 1
            if i == length - 1:
                num.append(count)
        else:
            num.append(count)
            count = 1

num = np.array(num)    
print(len(num))

X_val_Q = X_val_Q[:,:200]
X_val_A = X_val_A[:,:200]
X_val = np.concatenate((X_val_Q, X_val_A), axis=1)
X_val = np.expand_dims(X_val, axis=1)

Y_val = np.load('../data/numpy_array/validation_label.npy')

count = 0
zero_count = 0
for i in range(len(Y_val)):
    if Y_val[i] == 1:
        count += 1
    else:
        zero_count += 1
print(count, zero_count)

model_list = os.listdir('../model/net_model')

for i in model_list:
    model_name = '../model/net_model/' + i
    with CustomObjectScope({'Attention': Attention()}):
        model = load_model(model_name)

    #model.summary()

    #result = model.predict([X_val_Q, X_val_A])
    result = model.predict([X_val], 128)
    result = result.reshape(result.shape[0])

    mrr = MRR(result, Y_val)
    print(model_name, mrr)

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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 128

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

def merge():
    f1 = open('../data/result/1.txt', 'r')
    s1 = []
    while True:
        line = f1.readline()
        if not line:
            break
        s1.append(float(line))
    f1.close()
    s1 = np.array(s1)

    f2 = open('../data/result/2.txt', 'r')
    s2 = []
    while True:
        line = f2.readline()
        if not line:
            break
        s2.append(float(line))
    f2.close()
    s2 = np.array(s2)
    
    f3 = open('../data/result/3.txt', 'r')
    s3 = []
    while True:
        line = f3.readline()
        if not line:
            break
        s3.append(float(line))
    f3.close()
    s3 = np.array(s3)

    f4 = open('../data/result/4.txt', 'r')
    s4 = []
    while True:
        line = f4.readline()
        if not line:
            break
        s4.append(float(line))
    f4.close()
    s4 = np.array(s4)

    m1 = (s1 + s2) / 2
    m2 = (m1 + s3) / 2
    m3 = (m1 + s3) / 2
    m4 = (s1 + s2 + s3 + s4) / 4
    
    '''
    m1s = open('../data/result/m1.txt', 'w')
    for i in m1:
        m1s.write(str(i) + '\n')
    m1s.close()

    m2s = open('../data/result/m2.txt', 'w')
    for i in m2:
        m2s.write(str(i) + '\n')
    m2s.close()

    m3s = open('../data/result/m3.txt', 'w')
    for i in m3:
        m3s.write(str(i) + '\n')
    m3s.close()
    '''

    m4s = open('../data/result/score.txt', 'w')
    for i in m4:
        m4s.write(str(i) + '\n')
    m4s.close()
    


if __name__ == "__main__":
    '''
    X_val_Q = np.load('../data/numpy_array/validation_Q_index.npy')
    X_val_A = np.load('../data/numpy_array/validation_A_index.npy')
    X_val_Q = X_val_Q[:,:200]
    X_val_A = X_val_A[:,:200]
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

    #X_val = np.concatenate((X_val_Q, X_val_A), axis=1)
    #X_val = np.expand_dims(X_val, axis=1)

    Y_val = np.load('../data/numpy_array/validation_label.npy')

    count = 0
    zero_count = 0
    for i in range(len(Y_val)):
        if Y_val[i] == 1:
            count += 1
        else:
            zero_count += 1
    print(count, zero_count)

    base = '../model/abc_model/'
    model_list = os.listdir(base)

    X_val_Q0 = X_val_Q[61:,:200]
    X_val_A0 = X_val_A[61:,:200]
    X_val_Q1 = X_val_Q[:128,:200]
    X_val_A1 = X_val_A[:128,:200]

    for i in model_list:
        model_name = base + i
        #if i != 'model_07-0.79.hdf5':
        #    continue
        with CustomObjectScope({'Attention_matrix': Attention_matrix}, {'Attention_weight': Attention_weight}):
        #with CustomObjectScope({'Attention': Attention}, {'Attention_matrix': Attention_matrix}, {'Attention_weight': Attention_weight}):
            model = load_model(model_name)

        result0 = model.predict([X_val_Q0, X_val_A0], 128)
        result0 = result0.reshape(result0.shape[0])
        result1 = model.predict([X_val_Q1, X_val_A1], 128)
        result1 = result1.reshape(result1.shape[0])
        result = np.concatenate((result1[:61], result0))
        #result = model.predict([X_val], 128)
        #result = result.reshape(result.shape[0])
        
        file = open('../data/result/' + i + '_score.txt', 'w')
        for j in result:
            file.write(str(j) + '\n')
        file.close()
        
        mrr = MRR(result, Y_val)
        print(model_name, mrr)
    '''
    merge()

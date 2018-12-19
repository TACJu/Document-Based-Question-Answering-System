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

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
X_val_Q = X_val_Q[61:,:200]
X_val_A = X_val_A[61:,:200]
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
Y_val = Y_val[61:]

count = 0
zero_count = 0
for i in range(len(Y_val)):
    if Y_val[i] == 1:
        count += 1
    else:
        zero_count += 1
print(count, zero_count)

model_list = os.listdir('../model/new_net_model')

for i in model_list:
    model_name = '../model/new_net_model/' + i
    with CustomObjectScope({'Attention_matrix': Attention_matrix}, {'Attention_weight': Attention_weight}):
        model = load_model(model_name)

    result = model.predict([X_val_Q, X_val_A], 128)
    #result = model.predict([X_val], 128)
    result = result.reshape(result.shape[0])
    '''
    file = open('../data/result/' + i + '_score.txt', 'w')
    for j in result:
        file.write(str(j) + '\n')
    file.close()
    '''
    mrr = MRR(result, Y_val)
    print(model_name, mrr)

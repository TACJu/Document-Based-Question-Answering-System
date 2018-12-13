import numpy as np
import os
import keras
from keras import backend as K
from keras.models import load_model

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
X_train_Q = np.load('../data/numpy_array/train_Q_index.npy')
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
    print(model_name)
    model = load_model(model_name)

    #model.summary()

    result = model.predict([X_val_Q, X_val_A])
    result = result.reshape(result.shape[0])

    mrr = MRR(result, Y_val)
    print(mrr)

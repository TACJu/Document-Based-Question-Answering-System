import numpy as np
import os
import keras
from keras import backend as K
from keras.models import load_model

def MRR(pred, label):
    global num
    Q = len(num)
    mrr = 0
    index = 0
    for i in num:
        tmp_pred = pred[index:index + i]
        tmp_label = label[index:index + i]
        true_index = np.argmax(tmp_label)
        tmp_pred = np.argsort(-tmp_pred)
        rank = np.argwhere(tmp_pred == true_index)[0] + 1
        mrr += 1/rank
        index += i
    mrr /= Q
    return mrr
    
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#model = load_model('../model/model_01-0.96.hdf5')

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

count = 0
for i in num:
    count += i

print(count)

X_val = np.concatenate((X_val_Q, X_val_A), axis=1)
Y_val = np.load('../data/numpy_array/validation_label.npy')
print(len(Y_val))

count = 0
zero_count = 0
for i in range(len(Y_val)):
    if Y_val[i] == 1:
        count += 1
    elif Y_val[i] == 0:
        zero_count += 1
    else:
        print(Y_val[i])

'''
result = np.zeros((len(X_val)))
result = np.round(model.predict(X_val, 100))

tp = 0
tn = 0
fp = 0
fn = 0

for i in range(len(Y_val)):
    if Y_val[i] == 1 and result[i] == 1:
        tp += 1
    elif Y_val[i] == 1 and result[i] == 0:
        fn += 1
    elif Y_val[i] == 0 and result[i] == 1:
        fp += 1
    elif Y_val[i] == 0 and result[i] == 0:
        tn += 1
print(tp, tn, fp, fn)
'''
import numpy as np
import os
import keras
from keras import backend as K
from keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

model = load_model('../model/model_01-0.96.hdf5')

X_val_Q = np.load('../data/numpy_array/validation_Q_index.npy')
X_val_A = np.load('../data/numpy_array/validation_A_index.npy')
X_val = np.concatenate((X_val_Q, X_val_A), axis=1)
Y_val = np.load('../data/numpy_array/validation_label.npy')

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

# import numpy as np

# b = np.random.rand(10,1)
# print(b)
# m  = np.argmax(b, 0)
# print(m)

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.datasets import mnist


def load_data_1():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    SCALE_FACTOR = 255 # TRES IMPORTANT SINON OVERFLOW SUR EXP
    WIDTH = X_train.shape[1]
    HEIGHT = X_train.shape[2]
    # Input Layer neurons (784)
    INPUT_NEURONS = WIDTH * HEIGHT
    X_train = X_train.reshape(X_train.shape[0],INPUT_NEURONS).T / SCALE_FACTOR
    X_test = X_test.reshape(X_test.shape[0],INPUT_NEURONS).T  / SCALE_FACTOR
    return (X_train, Y_train), (X_test, Y_test)

(X_train, Y_train), (X_test, Y_test) = load_data_1()
print(X_train.shape)
print(Y_train.shape)
print(Y_train[:10])
print(Y_test[:10])
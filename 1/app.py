import os, io
# Set TF_ENABLE_ONEDNN_OPTS environment variable to 0
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import pickle
from keras.datasets import mnist
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import random

def ReLU(Z):
    return np.maximum(Z,0)

def derivative_ReLU(Z):
    return Z > 0

def softmax(Z):
    """Compute softmax values for each sets of scores in x."""
    exp = np.exp(Z - np.max(Z)) #le np.max(Z) evite un overflow en diminuant le contenu de exp
    return exp / exp.sum(axis=0)

def init_params(size):
    W1 = np.random.rand(10,size) - 0.5
    b1 = np.random.rand(10,1) - 0.5
    W2 = np.random.rand(10,10) - 0.5
    b2 = np.random.rand(10,1) - 0.5
    return W1,b1,W2,b2
    # W1 = np.random.normal(size=(10, 784)) * np.sqrt(1./(784))
    # b1 = np.random.normal(size=(10, 1)) * np.sqrt(1./10)
    # W2 = np.random.normal(size=(10, 10)) * np.sqrt(1./20)
    # b2 = np.random.normal(size=(10, 1)) * np.sqrt(1./(784))
    # return W1, b1, W2, b2

def forward_propagation(X,W1,b1,W2,b2):
    Z1 = W1.dot(X) + b1 #10, m
    A1 = ReLU(Z1) # 10,m
    Z2 = W2.dot(A1) + b2 #10,m
    A2 = softmax(Z2) #10,m
    return Z1, A1, Z2, A2

def one_hot(Y):
    ''' return an 0 vector with 1 only in the position correspondind to the value in Y'''
    one_hot_Y = np.zeros((Y.max()+1,Y.size)) #si le chiffre le plus grand dans Y est 9 ca fait 10 lignes
    one_hot_Y[Y,np.arange(Y.size)] = 1 # met un 1 en ligne Y[i] et en colonne i, change l'ordre mais pas le nombre
    return one_hot_Y

def backward_propagation(X, Y, A1, A2, W2, Z1, m):
    one_hot_Y = one_hot(Y)
    dZ2 = 2*(A2 - one_hot_Y) #10,m
    dW2 = 1/m * (dZ2.dot(A1.T)) # 10 , 10
    db2 = 1/m * np.sum(dZ2,1) # 10, 1
    dZ1 = W2.T.dot(dZ2)*derivative_ReLU(Z1) # 10, m
    dW1 = 1/m * (dZ1.dot(X.T)) #10, 784
    db1 = 1/m * np.sum(dZ1,1) # 10, 1

    return dW1, db1, dW2, db2

def update_params(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2):
    W1 -= alpha * dW1
    b1 -= alpha * np.reshape(db1, (10,1))
    W2 -= alpha * dW2
    b2 -= alpha * np.reshape(db2, (10,1))

    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y)/Y.size

def gradient_descent(X, Y, alpha, iterations):
    size , m = X.shape

    W1, b1, W2, b2 = init_params(size)
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward_propagation(X, Y, A1, A2, W2, Z1, m)

        W1, b1, W2, b2 = update_params(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2)   

        if (i+1) % int(iterations/10) == 0:
            print(f"Iteration: {i+1} / {iterations}")
            prediction = get_predictions(A2)
            print(f'{get_accuracy(prediction, Y):.3%}')
    return W1, b1, W2, b2

def make_predictions(X, W1 ,b1, W2, b2):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    predictions = get_predictions(A2)
    return predictions

def show_prediction(vect_X, label, W1, b1, W2, b2):
    """
    None => cree un nouvel axe de dimension 1, cela a pour effet de transposer X[:,index] 
    qui un np.array de dimension 1 (ligne) et qui devient un vecteur (colonne)
    ce qui correspond bien a ce qui est demande par make_predictions qui attend une matrice 
    dont les colonnes sont les pixels de l'image, la on donne une seule colonne
    """
    prediction = make_predictions(vect_X, W1, b1, W2, b2)
    print("Prediction: ", prediction)
    print("Label: ", label)
    WIDTH, HEIGHT, SCALE_FACTOR = 28, 28, 255
    current_image = vect_X.reshape((WIDTH, HEIGHT)) * SCALE_FACTOR

    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()



############## MAIN ##############

def load_data_1():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    SCALE_FACTOR = 255 # TRES IMPORTANT SINON OVERFLOW SUR EXP
    WIDTH = X_train.shape[1]
    HEIGHT = X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0],WIDTH*HEIGHT).T / SCALE_FACTOR
    X_test = X_test.reshape(X_test.shape[0],WIDTH*HEIGHT).T  / SCALE_FACTOR
    return X_train, Y_train, X_test, Y_test

def load_data_2():
    data = pd.read_csv('data/train.csv')
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data) # shuffle before splitting into dev and training sets
    data_test = data[0:1000].T
    Y_test = data_test[0]
    X_test = data_test[1:n]
    # X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])
    X_test = X_test / 255.

    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
    X_train = X_train / 255.
    # _,m_train = X_train.shape
    return X_train, Y_train, X_test, Y_test

def train(X_train, Y_train, iterations=200):
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.15, iterations)
    with open("trained_params.pkl","wb") as dump_file:
        pickle.dump((W1, b1, W2, b2),dump_file)

def process_image(image_path):
    # Load the image
    img = Image.open(image_path)
    #resize image to 28x28 pixels
    img = img.resize((28,28))
    #convert rgb to grayscale
    img = img.convert('L')
    img_array = np.array(img)
    img_array = img_array.reshape(28*28,1)
    # img_array = img_array.reshape(1,28,28,1)
    img_array = img_array/255.0
    img_array = 1 - img_array
    return img_array

if __name__ == '__main__':

    # loading data
    X_train, Y_train, X_test, Y_test = load_data_1()
    # X_train, Y_train, X_test, Y_test = load_data_2()
    """
    # training data and save the model
    timer_start = datetime.now()
    train(X_train, Y_train, 200) # W1, b1, W2, b2
    timer_end = datetime.now()
    difference = timer_end - timer_start
    print("The model has successfully trained in {:2f} seconds.".format(difference.total_seconds()))
    """
    # load the model
    with open("trained_params.pkl","rb") as dump_file:
        W1, b1, W2, b2 = pickle.load(dump_file)
    # predict
    # for i in range(1,10):
    #     img_array = process_image(image_path = f"digits/{i}.jpg")
    #     show_prediction(img_array, i, W1, b1, W2, b2)

    for i in range(20):
        index = random.randint(0, 1000)
        show_prediction(X_test[:, index,None], Y_test[index], W1, b1, W2, b2)
        # show_prediction(200 , X_test, Y_test, W1, b1, W2, b2)

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
    
    # dZ2 = (A2 - one_hot(Y))/Y.size
    # dW2 = dZ2.dot(A1.T)
    # db2 = np.sum(dZ2, axis = 1, keepdims = True)
    # dA1 = W2.T.dot(dZ2)
    # dZ1 = dA1* (Z1>0).astype(int)
    # dW1 = dZ1.dot(X.T)
    # db1 = np.sum(dZ1, axis = 1, keepdims = True)
    # return dW1, db1, dW2, db2

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

def get_loss(A2, Y, m):
    # m = A2.shape[1] Y.size
    return -np.sum(np.log(A2[Y, np.arange(m)]))/ m
    

def gradient_descent(X_train, Y_train, X_val, Y_val, alpha, iterations):
    size_train, m_train = X_train.shape
    size_val, m_val = X_val.shape
    W1, b1, W2, b2 = init_params(size_train)
    
    history = { "train": { "accuracy": [], "loss": []}, "validation": { "accuracy": [], "loss": []}}
    for i in range(iterations):
        # Training
        Z1_train, A1_train, Z2_train, A2_train = forward_propagation(X_train, W1, b1, W2, b2)
        # delta
        dW1_train, db1_train, dW2_train, db2_train = backward_propagation(X_train, Y_train, A1_train, A2_train, W2, Z1_train, m_train)
        W1, b1, W2, b2 = update_params(alpha, W1, b1, W2, b2, dW1_train, db1_train, dW2_train, db2_train)   

        # Validation
        Z1_val, A1_val, Z2_val, A2_val = forward_propagation(X_val, W1, b1, W2, b2)


        if (i + 1) % int(iterations / 10) == 0:
            train_prediction = get_predictions(A2_train)
            train_accuracy = get_accuracy(train_prediction, Y_train)
            train_loss = get_loss(A2_train, Y_train, m_train)
            val_loss = get_loss(A2_val, Y_val, m_val)
            val_prediction = get_predictions(A2_val)
            val_accuracy = get_accuracy(val_prediction, Y_val)
            history["validation"]["loss"].append(val_loss)
            history["validation"]["accuracy"].append(val_accuracy)
            history["train"]["loss"].append(train_loss)
            history["train"]["accuracy"].append(train_accuracy)

            print(f"Iteration: {i + 1} / {iterations}")
            print(f'Training Accuracy: {train_accuracy:.3%} | Training Loss: {train_loss:.4f}')
            print(f'Validation Accuracy: {val_accuracy:.3%} | Validation Loss: {val_loss:.4f}')
        

    show_evaluation(history, iterations)
    return W1, b1, W2, b2

def show_evaluation(history, iterations):
    train_accuracies, train_losses = history["train"]["accuracy"], history["train"]["loss"]
    validation_accuracies, validation_losses = history["validation"]["accuracy"], history["validation"]["loss"]
    # Create a single figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # range(iterations), 
    ax1.plot(train_accuracies, label='Training Accuracy')
    ax1.plot(validation_accuracies, label='Validation Accuracy')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    ax2.plot(train_losses, label='Training Loss')
    ax2.plot(validation_losses, label='Validation Loss')
    ax2.set_title('Loss')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.show()


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
    # Input Layer neurons (784)
    INPUT_NEURONS = WIDTH * HEIGHT
    X_train = X_train.reshape(X_train.shape[0],INPUT_NEURONS).T / SCALE_FACTOR
    X_test = X_test.reshape(X_test.shape[0],INPUT_NEURONS).T  / SCALE_FACTOR
    return (X_train, Y_train), (X_test, Y_test)

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
    return (X_train, Y_train), (X_test, Y_test)

def train(X_train, Y_train, X_test, Y_test, iterations=200):
    timer_start = datetime.now()
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, X_test, Y_test, 0.15, iterations)
    with open("trained_params.pkl","wb") as dump_file:
        pickle.dump((W1, b1, W2, b2),dump_file)
    timer_end = datetime.now()
    difference = timer_end - timer_start
    print("The model has successfully trained in {:2f} seconds.".format(difference.total_seconds()))
    return W1, b1, W2, b2

def load_model(): # file_path
    with open("trained_params.pkl","rb") as dump_file:
        W1, b1, W2, b2 = pickle.load(dump_file)
    return W1, b1, W2, b2

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
    (X_train, Y_train), (X_test, Y_test) = load_data_1()
    # (X_train, Y_train), (X_test, Y_test) = load_data_2()
    
    # # training data and save the model
    # train(X_train, Y_train, X_test, Y_test, 1000) # W1, b1, W2, b2
    
    # load the model
    W1, b1, W2, b2 = load_model()

    # predict
    # for i in range(1,10):
    #     img_array = process_image(image_path = f"digits/{i}.jpg")
    #     show_prediction(img_array, i, W1, b1, W2, b2)

    for i in range(20):
        index = random.randint(0, 1000)
        show_prediction(X_test[:, index,None], Y_test[index], W1, b1, W2, b2)
        # show_prediction(200 , X_test, Y_test, W1, b1, W2, b2)
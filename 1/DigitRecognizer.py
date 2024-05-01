import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import pickle
from keras.datasets import mnist
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import random


class DigitRecognizer(object):

    #############################################
    # constants
    #############################################

    WIDTH, HEIGHT, SCALE_FACTOR = 28, 28, 255
    ITERATIONS=200
    MODAL_FILE_NAME="trained_params.pkl"

    #############################################
    # constructor
    #############################################

    def __init__(self):
        pass

    #############################################
    # public methods
    #############################################

    def load_data(self):
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        SCALE_FACTOR = 255 # very IMPORTANT otherwise will obtain overflow on e (exponential) math function
        w = X_train.shape[1]
        h = X_train.shape[2]
        # Input Layer neurons (784)
        INPUT_NEURONS = w * h
        X_train = X_train.reshape(X_train.shape[0],INPUT_NEURONS).T / SCALE_FACTOR
        X_test = X_test.reshape(X_test.shape[0],INPUT_NEURONS).T  / SCALE_FACTOR
        return (X_train, Y_train), (X_test, Y_test)

    def train(self, X_train, Y_train, X_test, Y_test, iterations=ITERATIONS, file_path=MODAL_FILE_NAME):
        timer_start = datetime.now()
        self.weight_1, self.bias_1, self.weight_2, self.bias_2 = self.__gradient_descent(X_train, Y_train, X_test, Y_test, 0.15, iterations)
        with open(file_path,"wb") as dump_file:
            pickle.dump((self.weight_1, self.bias_1, self.weight_2, self.bias_2),dump_file)
        timer_end = datetime.now()
        difference = timer_end - timer_start
        print("The model has successfully trained in {:2f} seconds.".format(difference.total_seconds()))
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2

    def load_model(self, file_path=MODAL_FILE_NAME):
        with open(file_path,"rb") as dump_file:
            self.weight_1, self.bias_1, self.weight_2, self.bias_2 = pickle.load(dump_file)
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2

    def make_predictions(self, X):
        _, _, _, A2 = self.__forward_propagation(X)
        predictions = self.get_predictions(A2)
        return predictions
    
    #############################################
    # private methods
    #############################################

    def __init_params(self, size):
        W1 = np.random.rand(10,size) - 0.5
        b1 = np.random.rand(10,1) - 0.5
        W2 = np.random.rand(10,10) - 0.5
        b2 = np.random.rand(10,1) - 0.5
        self.weight_1, self.bias_1, self.weight_2, self.bias_2 = W1, b1, W2, b2
        return W1,b1,W2,b2
        # W1 = np.random.normal(size=(10, 784)) * np.sqrt(1./(784))
        # b1 = np.random.normal(size=(10, 1)) * np.sqrt(1./10)
        # W2 = np.random.normal(size=(10, 10)) * np.sqrt(1./20)
        # b2 = np.random.normal(size=(10, 1)) * np.sqrt(1./(784))
        # return W1, b1, W2, b2

    def __update_params(self, alpha, dW1, db1, dW2, db2):
        self.weight_1 -= alpha * dW1
        self.bias_1 -= alpha * np.reshape(db1, (10,1))
        self.weight_2 -= alpha * dW2
        self.bias_2 -= alpha * np.reshape(db2, (10,1))
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2

    def __forward_propagation(self, X):
        Z1 = self.weight_1.dot(X) + self.bias_1 #10, m
        A1 = DigitRecognizer.ReLU(Z1) # 10,m
        Z2 = self.weight_2.dot(A1) + self.bias_2 #10,m
        A2 = DigitRecognizer.softmax(Z2) #10,m
        return Z1, A1, Z2, A2

    def __backward_propagation(self, X, Y, A1, A2, Z1, m):
        one_hot_Y = one_hot(Y)
        dZ2 = 2*(A2 - one_hot_Y) #10,m
        dW2 = 1/m * (dZ2.dot(A1.T)) # 10 , 10
        db2 = 1/m * np.sum(dZ2,1) # 10, 1
        dZ1 = self.weight_2.T.dot(dZ2) * derivative_ReLU(Z1) # 10, m
        dW1 = 1/m * (dZ1.dot(X.T)) #10, 784
        db1 = 1/m * np.sum(dZ1,1) # 10, 1
        return dW1, db1, dW2, db2

    def __gradient_descent(self, X_train, Y_train, X_val, Y_val, alpha, iterations=ITERATIONS):
        size_train, m_train = X_train.shape
        size_val, m_val = X_val.shape
        self.__init_params(size_train)
        
        history = { "train": { "accuracy": [], "loss": []}, "validation": { "accuracy": [], "loss": []}}
        for i in range(iterations):
            # Training
            Z1_train, A1_train, Z2_train, A2_train = self.__forward_propagation(X_train)
            # delta
            dW1_train, db1_train, dW2_train, db2_train = self.__backward_propagation(X_train, Y_train, A1_train, A2_train, Z1_train, m_train)
            self.__update_params(alpha, dW1_train, db1_train, dW2_train, db2_train)   

            # Validation
            Z1_val, A1_val, Z2_val, A2_val = self.__forward_propagation(X_val)

            if (i + 1) % int(iterations / 10) == 0:
                train_prediction = DigitRecognizer.get_predictions(A2_train)
                train_accuracy = DigitRecognizer.get_accuracy(train_prediction, Y_train)
                train_loss = DigitRecognizer.get_loss(A2_train, Y_train, m_train)
                val_loss = DigitRecognizer.get_loss(A2_val, Y_val, m_val)
                val_prediction = DigitRecognizer.get_predictions(A2_val)
                val_accuracy = DigitRecognizer.get_accuracy(val_prediction, Y_val)
                history["validation"]["loss"].append(val_loss)
                history["validation"]["accuracy"].append(val_accuracy)
                history["train"]["loss"].append(train_loss)
                history["train"]["accuracy"].append(train_accuracy)

                print(f"Iteration: {i + 1} / {iterations}")
                print(f'Training Accuracy: {train_accuracy:.3%} | Training Loss: {train_loss:.4f}')
                print(f'Validation Accuracy: {val_accuracy:.3%} | Validation Loss: {val_loss:.4f}')
            
        self.show_evaluation(history, iterations)
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2

    def show_prediction(self, vect_X, label):
        prediction = self.make_predictions(vect_X)
        print("Prediction: ", prediction)
        print("Label: ", label)
        
        current_image = vect_X.reshape((DigitRecognizer.WIDTH, DigitRecognizer.HEIGHT)) * DigitRecognizer.SCALE_FACTOR

        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()


    #############################################
    # static methods
    #############################################

    @staticmethod
    def ReLU(Z):
        return np.maximum(Z,0)

    @staticmethod
    def derivative_ReLU(Z):
        return Z > 0

    @staticmethod
    def softmax(Z):
        exp = np.exp(Z - np.max(Z))
        return exp / exp.sum(axis=0)

    @staticmethod
    def one_hot(Y):
        one_hot_Y = np.zeros((Y.max()+1,Y.size))
        one_hot_Y[Y,np.arange(Y.size)] = 1
        return one_hot_Y

    @staticmethod
    def get_predictions(A2):
        return np.argmax(A2, 0)

    @staticmethod
    def get_accuracy(predictions, Y):
        return np.sum(predictions == Y)/Y.size

    @staticmethod
    def get_loss(A2, Y, m):
        # m = A2.shape[1] Y.size
        return -np.sum(np.log(A2[Y, np.arange(m)]))/ m
    
    @staticmethod
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

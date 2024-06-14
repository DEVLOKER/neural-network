import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pickle
from keras.datasets import mnist
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

class DigitRecognizer(object):

    #############################################
    # constants
    #############################################

    WIDTH, HEIGHT, SCALE_FACTOR = 28, 28, 255
    ITERATIONS=200 # epochs
    ALPHA=0.15 # learning_rate
    LEARNING_RATE = 0.01
    MODAL_FILE_NAME= os.path.join("results", "trained_params.pkl")
    TRAINING_HISTORY= os.path.join("results", "training_history.jpg")

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

    def train(self, X_train, Y_train, X_test, Y_test, iterations=ITERATIONS, alpha=ALPHA, file_path=MODAL_FILE_NAME):
        timer_start = datetime.now()
        gradiant = self.__gradient_descent(X_train, Y_train, X_test, Y_test, alpha, iterations)
        try:
            while True:
                val = next(gradiant)
                history, W1, b1, W2, b2 = val
                yield val
        except StopIteration:
            pass

        self.weight_1, self.bias_1, self.weight_2, self.bias_2 = W1, b1, W2, b2
        # with open(file_path,"wb") as dump_file:
        #     pickle.dump((W1, b1, W2, b2),dump_file)
        timer_end = datetime.now()
        difference = timer_end - timer_start
        print("The model has successfully trained in {:2f} seconds.".format(difference.total_seconds()))
        # self.show_evaluation(history)

    def load_model(self, file_path=MODAL_FILE_NAME):
        with open(file_path,"rb") as dump_file:
            W1, b1, W2, b2 = pickle.load(dump_file)
        self.weight_1, self.bias_1, self.weight_2, self.bias_2 = W1, b1, W2, b2
        return W1, b1, W2, b2

    def make_predictions(self, X):
        W1, b1, W2, b2 = self.weight_1, self.bias_1, self.weight_2, self.bias_2
        _, _, _, A2 = self.__forward_propagation(W1, b1, W2, b2, X)
        # digit, accuracy, predictions = DigitRecognizer.get_predictions(A2)
        A2 = A2.reshape(1, -1)[0] # A2_reshaped
        digit = np.argmax(A2) 
        accuracy = np.max(A2) * 100
        predictions = [(d, a*100) for d, a in enumerate(A2)]
        # return digit, accuracy, predictions
        return digit, accuracy, predictions #, A2
    
    #############################################
    # private methods
    #############################################

    def __init_params(self, size):
        self.weight_1 = np.random.rand(10,size) - 0.5
        self.bias_1 = np.random.rand(10,1) - 0.5
        self.weight_2 = np.random.rand(10,10) - 0.5
        self.bias_2 = np.random.rand(10,1) - 0.5
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2
        # DigitRecognizer.WIDTH * DigitRecognizer.HEIGHT
        # W1 = np.random.normal(size=(10, 784)) * np.sqrt(1./(784))
        # b1 = np.random.normal(size=(10, 1)) * np.sqrt(1./10)
        # W2 = np.random.normal(size=(10, 10)) * np.sqrt(1./20)
        # b2 = np.random.normal(size=(10, 1)) * np.sqrt(1./(784))
        # return W1, b1, W2, b2

    def __update_params(self, W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1    
        W2 = W2 - alpha * dW2  
        b2 = b2 - alpha * db2    
        return W1, b1, W2, b2

    def __forward_propagation(self, W1, b1, W2, b2, X):
        Z1 = W1.dot(X) + b1 #10, m
        A1 = DigitRecognizer.ReLU(Z1) # 10,m
        Z2 = W2.dot(A1) + b2 #10,m
        A2 = DigitRecognizer.softmax(Z2) #10,m
        return Z1, A1, Z2, A2

    def __backward_propagation(self, Z1, A1, Z2, A2, W1, W2, X, Y):
        _, m = X.shape
        one_hot_Y = DigitRecognizer.one_hot(Y)
        dZ2 = A2 - one_hot_Y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2)
        dZ1 = W2.T.dot(dZ2) * DigitRecognizer.derivative_ReLU(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1)
        return dW1, db1, dW2, db2

    def __gradient_descent(self, X_train, Y_train, X_val, Y_val, alpha=ALPHA, iterations=ITERATIONS):
        size_train, m_train = X_train.shape
        size_val, m_val = X_val.shape
        W1, b1, W2, b2 = self.__init_params(size_train)
        
        history = { "iterations": [], "train": { "accuracy": [], "loss": []}, "validation": { "accuracy": [], "loss": []}}
        for i in range(iterations):
            # Training
            Z1, A1, Z2, A2 = self.__forward_propagation(W1, b1, W2, b2, X_train)
            # delta
            dW1, db1, dW2, db2 = self.__backward_propagation(Z1, A1, Z2, A2, W1, W2, X_train, Y_train)
            W1, b1, W2, b2 = self.__update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

            # Validation
            Z1_val, A1_val, Z2_val, A2_val = self.__forward_propagation(W1, b1, W2, b2, X_val)

            if (i + 1) % int(iterations / 10) == 0:
                # digit, accuracy, predictions          train_prediction
                train_prediction  = DigitRecognizer.get_predictions(A2)#[1]
                train_accuracy = DigitRecognizer.get_accuracy(train_prediction, Y_train)
                train_loss = DigitRecognizer.get_loss(A2, Y_train, m_train)
                val_loss = DigitRecognizer.get_loss(A2_val, Y_val, m_val)
                val_prediction = DigitRecognizer.get_predictions(A2_val)#[1]
                val_accuracy = DigitRecognizer.get_accuracy(val_prediction, Y_val)
                history["iterations"].append(i+1)
                history["validation"]["loss"].append(val_loss)
                history["validation"]["accuracy"].append(val_accuracy)
                history["train"]["loss"].append(train_loss)
                history["train"]["accuracy"].append(train_accuracy)
                yield history, W1, b1, W2, b2
                # yield (i+1, train_accuracy, train_loss, val_accuracy, val_loss)
                # print(f"Iteration: {i + 1} / {iterations}")
                # print(f'Training Accuracy: {train_accuracy:.3%} | Training Loss: {train_loss:.4f}')
                # print(f'Validation Accuracy: {val_accuracy:.3%} | Validation Loss: {val_loss:.4f}')
            
        yield history, W1, b1, W2, b2
        # return W1, b1, W2, b2

    def show_evaluation(self, history, filename=TRAINING_HISTORY):
        # iterations = history["iterations"][-1]
        # Create a single figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        # Plot training & validation accuracy values
        ax1.plot(history["train"]['accuracy'])
        ax1.plot(history["validation"]["accuracy"])
        ax1.set_title('Model accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Validation'], loc='upper left')
        # Plot training & validation loss values
        ax2.plot(history["train"]["loss"])
        ax2.plot(history["validation"]["loss"])
        ax2.set_title('Model loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(filename)
        plt.show()

    #############################################
    # static methods
    #############################################

    @staticmethod
    def ReLU(Z): # The rectified linear unit 
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
    def process_image(img: Image):
        img = img.resize((DigitRecognizer.WIDTH,DigitRecognizer.HEIGHT)) # resize image to 28x28 pixels
        img = img.convert('L') # convert rgb to grayscale
        img_array = np.array(img)
        img_array = img_array.reshape(DigitRecognizer.WIDTH*DigitRecognizer.HEIGHT,1)
        img_array = img_array/DigitRecognizer.SCALE_FACTOR
        img_array = 1 - img_array
        return img_array



if __name__ == '__main__':
    digit_recognizer = DigitRecognizer()

    iterations = 10
    (X_train, Y_train), (X_test, Y_test) = digit_recognizer.load_data()
    training = digit_recognizer.train(X_train, Y_train, X_test, Y_test, iterations)
    try:
        while True:
            history, W1, b1, W2, b2 = next(training)
            train_accuracy = history["train"]["accuracy"][-1]
            train_loss = history["train"]["loss"][-1]
            val_accuracy = history["validation"]["accuracy"][-1]
            val_loss = history["validation"]["loss"][-1]
            i = history["iterations"][-1]
            text = f"""Iteration: {i} / {iterations}\nTraining Accuracy: {train_accuracy:.3%} | Training Loss: {train_loss:.4f}\nValidation Accuracy: {val_accuracy:.3%} | Validation Loss: {val_loss:.4f}"""
            print(text)
    except StopIteration:
        digit_recognizer.show_evaluation(history)

    # digit_recognizer.load_model()


    # # # predict
    # for i in range(1,1+1):
    #     img = Image.open(f"tmp/digits/{i}.jpg")  
    #     img_array = DigitRecognizer.process_image(img)
    #     # show_prediction(img_array, i, W1, b1, W2, b2)
    #     prediction = digit_recognizer.make_predictions(img_array)
    #     print("Prediction: ", prediction)
    #     print("Label: ", i)
    #     current_image = img_array.reshape((DigitRecognizer.WIDTH, DigitRecognizer.HEIGHT)) * DigitRecognizer.SCALE_FACTOR

    #     plt.gray()
    #     plt.imshow(current_image, interpolation='nearest')
    #     plt.show()

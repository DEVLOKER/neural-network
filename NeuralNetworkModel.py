import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pickle
from datetime import datetime
import numpy as np
from numpy import (ndarray)
from PIL import Image
from keras.datasets import mnist
from TrainingHistory import TrainingHistory


class NeuralNetworkModel(object):

    PREDICTED_CLASS = 10
    X_WIDTH = 28
    X_HEIGHT = 28
    INPUT_NEURONS_SIZE = 28 * 28 # Input Layer neurons (784)
    X_SCALE_FACTOR = 255
    LEARNING_RATE = 0.15 # 0.01
    EPOCHS=100 # iterations
    TARGET_ACCURANCY=0.9 # 
    MODAL_FILE_NAME= os.path.join("training", "trained_params.pkl")

    def __init__(self):
        self.training_history = TrainingHistory()

    def load_data(self):
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        scale_factor, input_size =  NeuralNetworkModel.X_SCALE_FACTOR, NeuralNetworkModel.INPUT_NEURONS_SIZE
        # Reshape the images from (28, 28) to (784,), Normalize the images to values between 0 and 1
        X_train = X_train.reshape(X_train.shape[0], input_size).T / scale_factor
        X_test = X_test.reshape(X_test.shape[0], input_size).T  / scale_factor
        return (X_train, Y_train), (X_test, Y_test)
    
    def init_params(self):
        self.W1 = np.random.normal(size=(10, 784)) * np.sqrt(1./(784))
        self.b1 = np.random.normal(size=(10, 1)) * np.sqrt(1./10)
        self.W2 = np.random.normal(size=(10, 10)) * np.sqrt(1./10*2)
        self.b2 = np.random.normal(size=(10, 1)) * np.sqrt(1./(784))
        return self.W1, self.b1, self.W2, self.b2

    def forward_propagation(self, X):
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = NeuralNetworkModel.relu(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = NeuralNetworkModel.softmax(Z2)
        return Z1, A1, Z2, A2
    
    def backward_propagation(self, X, Y, Z1, A1, Z2, A2):
        m = X.shape[1]
        dZ2 = A2 - NeuralNetworkModel.one_hot(Y)
        dW2 = np.dot(dZ2, A1.T) / m
        dB2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = self.W2.T.dot(dZ2) * (A1>0) # Z1>0
        dW1 = np.dot(dZ1, X.T) / m
        dB1 = np.sum(dZ1, axis=1, keepdims=True) / m
        return dW1, dB1, dW2, dB2

    def update_parameters(self, dW1, dB1, dW2, dB2, learning_rate=LEARNING_RATE):
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * dB1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * dB2
        return self.W1, self.b1, self.W2, self.b2

    def train(self, epochs=None, target_accurancy=None,learning_rate=LEARNING_RATE):
        # input train & test images
        (X_train, Y_train), (X_test, Y_test) = self.load_data()
        self.init_params()
        self.training_history.init(epochs)
        train_accurancy = 0
        epoch = 0
        timer_start = datetime.now()
        
        while (target_accurancy != None and train_accurancy < target_accurancy) or (target_accurancy == None and epochs != None and epoch < epochs):
            # training
            Z1, A1, Z2, A2 = self.forward_propagation(X_train)
            train_accurancy, train_loss = self.evaluate_model(A2, Y_train)
            dW1, dB1, dW2, dB2 = self.backward_propagation(X_train, Y_train, Z1, A1, Z2, A2)
            self.update_parameters(dW1, dB1, dW2, dB2, learning_rate)
            # validation
            Z1_val, A1_val, Z2_val, A2_val = self.forward_propagation(X_test)
            val_accurancy, val_loss = self.evaluate_model(A2_val, Y_test)
            # save histrory each 10 iterations
            if epoch % 1 == 0:
                self.training_history.append_history(epoch, train_accurancy, train_loss, val_accurancy, val_loss)
                yield self.training_history.get_last_history_epoch()
            epoch +=1
        
        timer_end = datetime.now()
        difference = timer_end - timer_start
        print(f"The model has successfully trained, {epoch} iterations in {difference.total_seconds():2f} seconds.")
        self.training_history.set_total_epochs(epoch)
        yield self.training_history.get_last_history_epoch()

    def make_predictions(self, X):
        _, _, _, output = self.forward_propagation(X)
        output = output.reshape(1, -1)[0]
        digit = np.argmax(output) 
        accuracy = np.max(output) * 100
        predictions = [(d, a*100) for d, a in enumerate(output)]
        return digit, accuracy, predictions

    def evaluate_model(self, A2, Y):
        accuracy = NeuralNetworkModel.compute_accuracy(A2, Y)
        loss = NeuralNetworkModel.compute_loss(A2, Y)
        return accuracy, loss

    def show_evaluation(self):
        self.training_history.show_evaluation()


    #############################################
    # static methods
    #############################################

    @staticmethod
    def compute_loss(A2, Y):
        # m = Y.shape[1]
        m = Y.size
        return -np.sum(np.log(A2[Y, np.arange(m)]))/ m
        # log_probs = np.multiply(np.log(A2), Y)
        # cost = -np.sum(log_probs) / m
        # return np.squeeze(cost)

    @staticmethod
    def compute_accuracy(A2, Y):
        predictions = np.argmax(A2, axis=0)
        return np.sum(predictions == Y)/Y.size
        # labels = np.argmax(Y, axis=0)
        # accuracy = np.mean(predictions == labels)
        # return accuracy
            
    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def softmax(Z):
        exp_z = np.exp(Z - np.max(Z))
        return exp_z / exp_z.sum(axis=0, keepdims=True)

    @staticmethod
    def one_hot(Y):
        one_hot_Y = np.zeros((Y.max()+1,Y.size))
        one_hot_Y[Y,np.arange(Y.size)] = 1
        return one_hot_Y
        
    @staticmethod
    def process_image(img: Image): # ndarray : NDArray
        # resize image to 28x28 pixels
        img = img.resize((NeuralNetworkModel.X_WIDTH, NeuralNetworkModel.X_HEIGHT)) 
        # convert rgb to grayscale
        img = img.convert('L')
        img_array = np.array(img)
        img_array = img_array.reshape(NeuralNetworkModel.INPUT_NEURONS_SIZE, 1)
        img_array = img_array/NeuralNetworkModel.X_SCALE_FACTOR
        img_array = 1 - img_array
        return img_array

    def load_model(self, file_path=MODAL_FILE_NAME):
        with open(file_path,"rb") as dump_file:
            model_parameters = pickle.load(dump_file)
            self.W1 = model_parameters['W1']
            self.b1 = model_parameters['B1']
            self.W2 = model_parameters['W2']
            self.b2 = model_parameters['B2']
        # return self.W1, self.b1, self.W2, self.b2
    
    def save_model(self, file_path=MODAL_FILE_NAME):
        model_parameters = {'W1': self.W1, 'B1': self.b1, 'W2': self.W2, 'B2': self.b2}
        with open(file_path,"wb") as dump_file:
            pickle.dump(model_parameters,dump_file)


#############################################
# test in main function
#############################################
if __name__ == '__main__':
    model = NeuralNetworkModel()
    
    # # model.train(epochs=100, learning_rate=0.15)
    training = model.train(epochs=10, target_accurancy=0.9, learning_rate=0.15)
    try:
        while True:
            text, epoch, train_accurancy, train_loss, val_accurancy, val_loss = next(training)
            # print(text)
    except StopIteration:
        model.save_model()
        model.show_evaluation()

    # model.load_model()

    for i in range(0,9+1):
        img = Image.open(f"tmp/digits/{i}.jpg")  
        img_array = NeuralNetworkModel.process_image(img)
        # show_prediction(img_array, i, W1, b1, W2, b2)
        digit, accuracy, predictions = model.make_predictions(img_array)

        print(f"Label: {i} <=> Digit: {digit}")



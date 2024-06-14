import os
import pickle
import numpy as np
from keras.datasets import mnist
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class NeuralNetworkModel(object):

    PREDICTED_CLASS = 10
    X_WIDTH = 28
    X_HEIGHT = 28
    INPUT_NEURONS_SIZE = 28 * 28 # Input Layer neurons (784)
    X_SCALE_FACTOR = 255
    LEARNING_RATE = 0.01
    EPOCHS=10 # iterations
    MODAL_FILE_NAME= os.path.join("results", "trained_params.pkl")

    def __init__(self):
        pass

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

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def softmax(Z):
        exp_z = np.exp(Z - np.max(Z))
        return exp_z / exp_z.sum(axis=0, keepdims=True)

    def forward_propagation(self, X):
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = NeuralNetworkModel.relu(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = NeuralNetworkModel.softmax(Z2)
        return Z1, A1, Z2, A2

    def compute_loss(self, A2, Y):
        # m = Y.shape[1]
        m = Y.size
        # log_probs = np.multiply(np.log(A2), Y)
        # cost = -np.sum(log_probs) / m
        # return np.squeeze(cost)
        return -np.sum(np.log(A2[Y, np.arange(m)]))/ m

    def compute_accuracy(self, A2, Y):
        predictions = np.argmax(A2, axis=0)
        # labels = np.argmax(Y, axis=0)
        # accuracy = np.mean(predictions == labels)
        # return accuracy

        return np.sum(predictions == Y)/Y.size
            


    @staticmethod
    def one_hot(Y):
        one_hot_Y = np.zeros((Y.max()+1,Y.size))
        one_hot_Y[Y,np.arange(Y.size)] = 1
        return one_hot_Y
    
    def backward_propagation(self, X, Y, Z1, A1, Z2, A2):
        m = X.shape[1]
        dZ2 = A2 - NeuralNetworkModel.one_hot(Y)
        dW2 = np.dot(dZ2, A1.T) / m
        dB2 = np.sum(dZ2, axis=1, keepdims=True) / m
        # dA1 = np.dot(self.W2.T, dZ2)
        # dZ1 = dA1 * (A1 > 0)
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

    def train(self, epochs=EPOCHS, learning_rate=LEARNING_RATE):
        (X_train, Y_train), (X_test, Y_test) = self.load_data()
        # print(Y_train.size, Y_train.shape)
        self.init_params()
        accuracy = 0
        epoch = 0
        # while accuracy < 0.9:
        while epoch < epochs and accuracy < 0.9:
        # for epoch in range(epochs):
            Z1, A1, Z2, A2 = self.forward_propagation(X_train)
            cost = self.compute_loss(A2, Y_train)
            accuracy = self.compute_accuracy(A2, Y_train)
            dW1, dB1, dW2, dB2 = self.backward_propagation(X_train, Y_train, Z1, A1, Z2, A2)
            self.update_parameters(dW1, dB1, dW2, dB2, learning_rate)
            # self.W1, self.b1, self.W2, self.b2
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Cost: {cost:.3f}, Accuracy: {accuracy:.3f}')
            epoch +=1






    def set_params(self, W1, b1, W2, b2):
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2   

    def get_params(self):
        return self.W1, self.b1, self.W2, self.b2  

    def set_input_layer(self, X):
        self.X = X

    def load_model(self, file_path=MODAL_FILE_NAME):
        with open(file_path,"rb") as dump_file:
            model_parameters = pickle.load(dump_file)
            self.W1 = model_parameters['W1']
            self.b1 = model_parameters['B1']
            self.W2 = model_parameters['W2']
            self.b2 = model_parameters['B2']
        return self.get_params()
    
    def save_model(self, file_path=MODAL_FILE_NAME):
        model_parameters = {'W1': self.W1, 'B1': self.b1, 'W2': self.W2, 'B2': self.b2}
        with open(file_path,"wb") as dump_file:
            pickle.dump(model_parameters,dump_file)


#############################################
# test in main function
#############################################
if __name__ == '__main__':
    model = NeuralNetworkModel()
    model.train(epochs=100, learning_rate=0.15)
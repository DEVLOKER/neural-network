import os
import pickle


class NeuralNetworkModel(object):

    MODAL_FILE_NAME= os.path.join("results", "trained_params.pkl")  

    def __init__(self):
        pass

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

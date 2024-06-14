import os
import matplotlib.pyplot as plt

class TrainingHistory(object):

    TRAINING_HISTORY= os.path.join("results", "training_history.jpg")

    def __init__(self, total_epochs):
        self.init(total_epochs)

    def init(self, total_epochs):
        self.total_epochs = total_epochs
        self.epochs = []
        self.training_accuracy = []
        self.training_loss = []
        self.validation_accuracy = []
        self.validation_loss = []

    def append_history(self, epoch, training_accuracy, training_loss, validation_accuracy, validation_loss):
        self.epochs.append(epoch)
        self.training_accuracy.append(training_accuracy)
        self.training_loss.append(training_loss)
        self.validation_accuracy.append(validation_accuracy)
        self.validation_loss.append(validation_loss)
        text = f"""Iteration: {epoch} / {self.total_epochs}\nTraining Accuracy: {training_accuracy:.3%} | Training Loss: {training_loss:.4f}\nValidation Accuracy: {validation_accuracy:.3%} | Validation Loss: {validation_loss:.4f}"""
        print(text)

    def get_epochs(self):
        return self.epochs
    
    def get_training_accuracy(self):
        return self.training_accuracy
    
    def get_training_loss(self):
        return self.training_loss
    
    def get_validation_accuracy(self):
        return self.validation_accuracy
    
    def get_validation_loss(self):
        return self.validation_loss
    
    def get_history(self):
        return self.epochs, self.training_accuracy, self.training_loss, self.validation_accuracy, self.validation_loss
    

    def get_history_by_epoch(self, epoch):
        return self.epochs[epoch], self.training_accuracy[epoch], self.training_loss[epoch], self.validation_accuracy[epoch], self.validation_loss[epoch]
    
    def get_last_history_epoch(self):
        return self.get_history_by_epoch(-1)
    

    def show_evaluation(self, filename=TRAINING_HISTORY):
        # Create a single figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot training & validation accuracy values
        ax1.plot(self.training_accuracy)
        ax1.plot(self.validation_accuracy)
        ax1.set_title('Model accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Training', 'Validation'], loc='upper left')
        
        # Plot training & validation loss values
        ax2.plot(self.training_loss)
        ax2.plot(self.validation_loss)
        ax2.set_title('Model loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Training', 'Validation'], loc='upper left')
        
        plt.savefig(filename)
        plt.show()
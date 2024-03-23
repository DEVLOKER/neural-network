import os, io
# Set TF_ENABLE_ONEDNN_OPTS environment variable to 0
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np 
from keras.datasets import mnist
import matplotlib.pyplot as plt
from random import shuffle
from NN import NN
from ReLU import ReLU
from Linear import Linear


# Defining the update function (SGD with momentum)
def update_params(velocity, params, grads, learning_rate=0.01, mu=0.9):
    for v, p, g, in zip(velocity, params, reversed(grads)):
        for i in range(len(g)):
            v[i] = mu * v[i] + learning_rate * g[i]
            p[i] -= v[i]
            # print('Max gradient value:',np.amax(v[i]))
            # print('Gradient shape:',v[i].shape)

# Defining a function which gives us the minibatches (both the datapoint and 
# the corresponding label)
# get minibatches
def minibatch(X, y, minibatch_size):
    n = X.shape[0]
    minibatches = []
    permutation = np.random.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]

    for i in range(0, n , minibatch_size):
        X_batch = X[i:i + minibatch_size, :]
        y_batch = y[i:i + minibatch_size, ]
        minibatches.append((X_batch, y_batch))
        
    return minibatches

# The traning loop
def train(net, X_train, y_train, minibatch_size, epoch, learning_rate, mu=0.9, X_val=None, y_val=None):
    val_loss_epoch = []
    minibatches = minibatch(X_train, y_train, minibatch_size)
    minibatches_val = minibatch(X_val, y_val, minibatch_size)

    for i in range(epoch):
        loss_batch = []
        val_loss_batch = []
        velocity = []
        for param_layer in net.params:
            p = [np.zeros_like(param) for param in list(param_layer)]
            velocity.append(p)
            
        # iterate over mini batches
        for X_mini, y_mini in minibatches:
            loss, grads = net.train_step(X_mini, y_mini)
            loss_batch.append(loss)
            update_params(velocity, net.params, grads, learning_rate=learning_rate, mu=mu)

        for X_mini_val, y_mini_val in minibatches_val:
            val_loss, _ = net.train_step(X_mini, y_mini)
            val_loss_batch.append(val_loss)
        
        # accuracy of model at end of epoch after all mini batch updates
        m_train = X_train.shape[0]
        m_val = X_val.shape[0]
        y_train_pred = np.array([], dtype="int64")
        y_val_pred = np.array([], dtype="int64")
        y_train1 = []
        y_vall = []
        for i in range(0, m_train, minibatch_size):
            X_tr = X_train[i:i + minibatch_size, : ]
            y_tr = y_train[i:i + minibatch_size,]
            y_train1 = np.append(y_train1, y_tr)
            y_train_pred = np.append(y_train_pred, net.predict(X_tr))

        for i in range(0, m_val, minibatch_size):
            X_va = X_val[i:i + minibatch_size, : ]
            y_va = y_val[i:i + minibatch_size,]
            y_vall = np.append(y_vall, y_va)
            y_val_pred = np.append(y_val_pred, net.predict(X_va))
            
        train_acc = check_accuracy(y_train1, y_train_pred)
        val_acc = check_accuracy(y_vall, y_val_pred)

        mean_train_loss = sum(loss_batch) / float(len(loss_batch))
        mean_val_loss = sum(val_loss_batch) / float(len(val_loss_batch))
        
        val_loss_epoch.append(mean_val_loss)
        print("Loss = {0} | Training Accuracy = {1} | Val Loss = {2} | Val Accuracy = {3}".format(mean_train_loss, train_acc, mean_val_loss, val_acc))
    return net

# Checking the accuracy of the model
def check_accuracy(y_true, y_pred):
    return np.mean(y_pred == y_true)

def main():
        
    (train_features, train_targets), (test_features, test_targets) = mnist.load_data()

    train_features = train_features.reshape(60000, 784)
    print(train_features.shape)
    test_features = test_features.reshape(10000, 784)
    print(test_features.shape)


    # # normalize inputs from 0-255 to 0-1
    train_features = train_features / 255.0
    test_features = test_features / 255.0

    print(train_targets.shape)
    print(test_targets.shape)

    X_train = train_features
    y_train = train_targets

    X_val = test_features
    y_val = test_targets

    # # visualizing the first 10 images in the dataset and their labels
    # plt.figure(figsize=(10, 1))
    # for i in range(10):
    #     plt.subplot(1, 10, i+1)
    #     plt.imshow(X_train[i].reshape(28, 28), cmap="gray")
    #     plt.axis('off')
    # plt.show()
    # print('label for each of the above image: %s' % (y_train[0:10]))
    
    # np.random.seed(3)


    ## input size
    input_dim = X_train.shape[1]

    ## hyperparameters
    iterations = 10
    learning_rate = 0.1
    hidden_nodes = 32
    output_nodes = 10

    ## define neural net
    nn = NN()
    nn.add_layer(Linear(input_dim, hidden_nodes))
    nn.add_layer(ReLU())
    nn.add_layer(Linear(hidden_nodes, output_nodes))

    nn = train(
        nn, 
        X_train, 
        y_train, 
        minibatch_size=200, 
        epoch=10, 
        learning_rate=learning_rate, 
        X_val=X_val, 
        y_val=y_val
    )

    # fprop a single image and showing its prediction
    plt.imshow(X_val[0].reshape(28,28), cmap='gray')
    # Predict Scores for each class
    prediction = nn.predict_scores(X_val[0])[0]
    print ("Scores")
    print (prediction)
    np.argmax(prediction)
    predict_class = nn.predict(X_val[0])[0]
    print(predict_class)


if __name__ == '__main__':
    main()


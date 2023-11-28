# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP9. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):
        in_size -> h -> out_size , where  1 <= h <= 256
        
        We recommend setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        self.linear1 = nn.Linear(2883, 512)
        self.linear3 = nn.Linear(512, 64)
        self.linear2 = nn.Linear(64, 4)
        self.relu = nn.ReLU()
        
        self.loss_fn = loss_fn
        self.lrate = lrate
        self.in_size = in_size
        self.out_size = out_size
    

    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        # raise NotImplementedError("You need to write this part!")
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        pridict = self.forward(x)
        loss = self.loss_fn(self.criterion, pridict, y)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # raise NotImplementedError("You need to write this part!")
        return loss.item()

def train(model, lr, trainLoader):
    for i, data in enumerate(trainLoader, 0):
        images, labels = data
        loss = model.step(images, labels)
        # loss = loss.astype(float)
        # print(type(loss))
    return loss

def test(model, dataLoader):
    label_lst = []
    for i, data in enumerate(dataLoader):
        image = data 
        predict = model(image)
        _, label = torch.max(predict, dim=1)
        label_lst.append(label.item())  
        
    return label_lst
def data_std(data):
    mean, std = torch.std_mean(data, dim=1, keepdim=True)
    data = (data - mean ) / std
    return data 

def combine_data(image, label):
    train_data = data_std(image)
    data = []
    for i in range(len(label)):
        data.append((train_data[i], label[i]))
    return data
def loss_func(criterion, y_hat, y):
    loss = criterion(y_hat, y)
    return loss

# def confusion_mat()

def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ 
    Make NeuralNet object 'net'. Use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method *must* work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of floats containing the total loss at the beginning and after each epoch.
        Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    channel_out, channel_in = train_set.shape
    lrate = 4.5e-3
    model = NeuralNet(lrate, loss_fn=loss_func, in_size=channel_in, out_size=channel_out)
    train_data = combine_data(train_set, train_labels)
    trainLoader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
    dev_set = data_std(dev_set)
    testLoader = torch.utils.data.DataLoader(dev_set, batch_size=1, shuffle= False)
    model.optimizer = optim.SGD(model.parameters(), lr=lrate, momentum=0.9)
    # TODO load data
    losses = []
    for epoch in range(epochs):
        loss = train(model, lrate, trainLoader)
        losses.append(loss)
    yhats = test(model, testLoader)
    yhats = np.array(yhats)
    return losses ,yhats,model

# strela neural net package
#
# This program is an implementation of a neural net. This file can be
# imported and instances of a strela_net (neural net) can be created,
# trained, and used for classification.
#
# strela is named after the first Soviet mainframe computer
# https://en.wikipedia.org/wiki/Strela_computer
#
# Table of Contents
#
# Revision History
# 07/14/18    Tim Liu    wrote function and class descriptors
# 07/15/18    Tim Liu    wrote outline for training and back propagation
# 07/15/18    Tim Liu    completed class initialization function
# 07/17/18    Tim Liu    wrote predict and activation methods
# 07/26/18    Tim Liu    modified classify to allow for multi-dimensional y
#                        predictions
# 07/26/18    Tim Liu    wrote training method - needs to be checked
# 07/28/18    Tim Liu    updated activation to be recursive and accept arbitrary
#                        input dimensions
# 08/13/18    Tim Liu    cleaned up print statements in training method for
#                        debugging
# 08/13/18    Tim Liu    removed unnecessary square in derivative of
#                        error function in training method
# 08/13/18    Tim Liu    changed dimension of x_l in final layer to n_outputs
#                        + 1 to ensure consistent indexing
# 08/13/18    Tim Liu    commented out special x_l case for final layer
# 08/22/18    Tim Liu    modified x_l matrix so final layer has 0th index output
# 08/23/18    Tim Liu    eliminated bias term in all layers
# 08/23/18    Tim Liu    restored bias term in first layer; no bias in other layers
# 09/03/18    Tim Liu    unused 0th terms for biases removed from arrays
#                        removed debugging print statements
# 09/09/18    Tim Liu    modified train to make multiple passes through data
# 09/10/18    Tim Liu    added method set_weight
# 09/19/18    Tim Liu    added methods self.act and self.loss to strela_net
#                        allowing for different activation and loss functions
# 09/29/18    Tim Liu    added l2 regularization term to weight update

import numpy as np
import strela_helpers as sh

class strela_net():
    '''this class is a neural net that can be used for classifcation and
       prediction. The class does not inherit from a superclass'''
    def __init__(self, inputs, outputs, h_layers, h_layers_d, \
        lr = 0.01, loss = "squared", act = "tanh", reg = 0, softmax = False):
        '''initializes the neural net. Creates a matrix for the weights
        and randomly initializes them
        inputs: inputs - number of inputs to the net
                outputs - number of output heads from the net
                h_layers - number of hidden layers
                h_layers_d - dimension of each hidden layer

        outputs: none'''

    
        # list of activations and losses currently supported
        self.supported_act = ["tanh"]
        self.supported_loss = ["squared", "cce"]
        
        # learning parameters
        self.lr = lr               # learning rate
        self.reg = reg             # regularization - currently not supported
        self.softmax = softmax # apply softmax
        
        # parameters that change during training
        self.weights = [] # current weights
        self.delta_l = [] # 2d array of delta from most recent BP
        self.x_l = []     # 2d array of the x at each step

        # parameters for loss function and activation
        self.act = act    # activation function
        self.loss = loss  # loss function
        
        # parameters describing shape of the neural net
        self.n_inputs = inputs       # no. inputs (excludes x0 = 1)
        self.n_outputs = outputs     # no. output heads
        self.h_layers = h_layers     # no. hidden layers
        self.h_layers_d = h_layers_d # no. nodes/layer (excluding zeroth coor)
        self.total_layers = h_layers + 1  # index of last layer

        # check that inputs are valid
        if self.check_inputs():
            # inputs not valid - exit without completing init
            return
        
        # arrays describing shape of the neural net
        # total number of layers (L) is no. hidden layers + 1 input layer + 
        # 1 output layer; 1 added to total layers to include 0 indexed
        # input "layer"
        
        # inputs to each layer - init all to dimension of hidden layers
        self.i_l = [self.h_layers_d for i in range(self.total_layers + 1)] 
        # outputs from each layer - init all to dimension of hidden layers
        self.j_l = [self.h_layers_d for j in range(self.total_layers + 1)]

        # fill in the special cases
        self.i_l[0] = self.n_inputs + 1              # first input has bias term
        self.j_l[0] = 0                              # output indexing starts at 1
        self.j_l[self.total_layers] = self.n_outputs # output layer

        
        # set up the weights, delta, and x arrays to be the correct shape
        for l in range(self.total_layers + 1):

            # randomly initialize weights - weight indexing starts at 1
            self.weights.append\
            (0.5 * (np.random.rand(self.i_l[l-1], self.j_l[l]) - 0.5))

            # set up the dimensions of the x outputs
            if l == 0:
                # "output" of zeroeth layer is the x point being trained on
                self.x_l.append(np.zeros((self.n_inputs + 1, 1)))
            elif l == self.total_layers:
                # nodes in final layer (nuber of output heads)
                self.x_l.append(np.zeros((self.n_outputs, 1)))
            else:
                # nodes in the hidden layers
                self.x_l.append(np.zeros((self.h_layers_d, 1)))
                
            # set up dimensions of delta
            self.delta_l.append(np.zeros((self.j_l[l], 1)))

        # set up the x0 bias term in the zeroth layer
        self.x_l[0][0][0] = 1

        # some info about initialization
        print("Strela Net initialized")
        print(self.__repr__())
                    
        return
    
    def train(self, x_all, y_all, epochs = 25):
        '''trains the neural net using stochastic gradient descent
        inputs: x_all - 2d list like object of all x inputs 
                y_all - 2d array of correct classifications
                epochs - number of passes through data
        outputs: none'''

        # convert x and y to 2D np arrays
        x_all = sh.make_2d(x_all)  
        y_all = sh.make_2d(y_all)

        # total number of points to train on
        total_train = len(x_all) * epochs
        # number of points trained on so far
        current_train = 0
        
        for i in range(epochs):
            # perform multiple passes through the data
            order = np.arange(len(x_all))
            # array of indexes in random order - used for SGD   
            np.random.shuffle(order)
            # train on the points in a random order
            for n in order:
                # generate the prediction; pass 2D array pull off first
                # element to get 1D array
                y_pre = self.predict([x_all[n]])[0]
                # calculate the sigma of the last layer - layer L
                self.delta_l[self.total_layers] = self.dx_ds(y_pre)\
                 * self.de_dx(y_pre, y_all[n])

                # go backwards in L and calculate the deltas
                for l in reversed(range(2, self.total_layers + 1)):
                    # back-propagate the deltas
                    self.delta_l[l-1] = self.dx_ds(self.x_l[l-1]) * \
                                        np.dot(self.weights[l], self.delta_l[l])
                # now update the weights
                for l in range(1, self.total_layers + 1):
                    # update weights of each layer
                    self.weights[l] -=\
                        self.lr * np.dot(self.x_l[l-1], np.transpose(self.delta_l[l])) \
                        + self.reg * self.weights[l]  # regularization term
                current_train += 1
                # change divisor to tune how often progress bar prints out
                if current_train % (total_train/20) == 0:
                    print("%.1f%% of training complete" %(current_train/total_train * 100))

        return
    
    def predict(self, x_all):
        '''generates a prediction based on the current weights of the net.
        the result of each for the most recent point is stored in self.x_l
        inputs: x_all - 2d array of each x point to generate a prediction on
        outputs: y_pre - 2d array of predicted y_coordinates'''
        
        # set up array of result of predictions
        y_pre = np.zeros((len(x_all), self.n_outputs))
        
        for n in range(len(x_all)):
            # reshape the input into a 2D n x 1 array
            self.x_l[0][1:] = sh.make_2d(x_all[n])
            for l in range(1, self.total_layers + 1):            
                # mutliply weights by x of previous layer to get intermediate s
                s_j = np.transpose(np.dot(np.transpose(self.x_l[l-1]), self.weights[l]))
                # apply activation to get x_l               
                self.x_l[l] = self.activation(s_j)
            # prediction is the output of the final layer
            # reshape to be one dimensional array output
            y_pre[n] = (self.x_l[self.total_layers]).reshape(self.n_outputs,)
            # apply softmax if set to do so
            if self.softmax:
                y_pre[n] = sh.apply_softmax(y_pre[n])
        # return array of predictions
        return y_pre

    def activation(self, s):
        '''activation function. Recursively 
         applies the activation function to a np object (array or float)
         of any shape elementwise. The return value should have the same
         dimensions as the argument
         inputs: s - object to apply activation function to
         outputs: theta - new object with activation applied'''
        theta = []
        # base case - check it's a float/int
        if np.shape(s) == ():
            # implementation of different activation functions
            if self.act == "tanh":
                return(np.exp(s) - np.exp(-s))/(np.exp(s) + np.exp(-s))

        else:
             # otherwise iterate through and perform recursive call
            for x in s:
                theta.append(self.activation(x))
        return np.array(theta)

    def de_dx(self, y_pre, y_n):
        '''returns the derivative de over dx, the derivative of the 
        error function with respect to x.
        inputs: y_pre - predicted y value
                y_n - true y value'''


        # de over dx for different loss functions
        if self.loss == "squared":
            # de/dx of squared loss: e = (y-yn) ** 2
            de_dx = 2 * (y_pre - y_n)
        if self.loss == "cce":
            # de/dx of cross-entropy: -(ynlog(y) + (1-yn)log(1-y))
            de_dx = -1 * y_n/y_pre + (1-y_n)/(1-y_pre)

        return sh.make_2d(de_dx)

    def dx_ds(self, x_l):
        '''calculates the derivative dx over ds, equivalent to the
        derivative of the activation function w/ respect to the
        input s
        inputs: x_l - output of activation for a given layer'''

        # dx over ds for different activation functions
        if self.act == "tanh":
            # derivative of tanh activation function
            dx_ds = (1 - x_l**2)

        return sh.make_2d(dx_ds)

    def set_weights(self, layer, new_weights):
        '''set the weights of a specific layer. If the passed weights
        do not match the dimensions of the layer, then an error is
        printed and weights are not changed
        inputs: layer - index of the layer to change weights (starts at 1)
                weights - numpy array of weights'''
        
        if layer > self.total_layers:
            print("Passed layer %d; total layers %d" %(layer, self.total_layers))
            return
        # check that dimensions match
        if np.shape(self.weights[layer]) != np.shape(new_weights):
            print("Dimension of passed weights do not match!")
            return

        self.weights[layer] = new_weights
        return

    def check_inputs(self):
        '''error checking function that checks if the combination of inputs
        to the neural net are allowed. Returns if True if there is an error'''

        error = False  # optimistically assume no error

        # check that loss and activation functions are recognized
        if self.loss not in self.supported_loss:
            print("Loss function not recognized; must be in: ")
            print(self.supported_loss)
            error = True
        if self.act not in self.supported_act:
            print("Activation function not recognized; must be in: ")
            print(self.supported_act)
            error = True

        return error


    def show_weights(self):
        '''prints the weights of the neural net in pretty form'''
        for l in range(1, self.total_layers + 1):
            print("Layer: ", l)
            print(self.weights[l])
        
        return

    def save_weights(self):
        '''saves the weights to a text file and a pickle file'''

        print("Sorry! This method isn't done yet :(")
        return

    def __repr__(self):
        '''prints info about the instance of strela'''
        output_string = 'Instance of strela_net class\n'
        output_string += "%d inputs, %d outputs\n" %(self.n_inputs, self.n_outputs)
        output_string += "%d hidden layers, %d nodes per hidden layer\n"\
        %(self.h_layers, self.h_layers_d)
        output_string += "Regularization lambda: %f\n" %(self.reg)
        output_string += "Activation function: %s\n" %(self.act)
        output_string += "Loss function: %s\n" %(self.loss)

        return output_string











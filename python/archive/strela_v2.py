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

import numpy as np

class strela_net():
    '''this class is a neural net that can be used for classifcation and
       prediction. The class does not inherit from a superclass'''
    def __init__(self, inputs, outputs, h_layers, h_layers_d, lr = 0.01):
        '''initializes the neural net. Creates a matrix for the weights
        and randomly initializes them
        inputs: inputs - number of inputs to the net
                outputs - number of output heads from the net
                h_layers - number of hidden layers
                h_layers_d - dimension of each hidden layer

        outputs: none'''
        
        # learning parameters
        self.lr = lr      # learning rate        
        
        # parameters that change during training
        self.weights = [] # current weights
        self.delta_l = [] # 2d array of delta from most recent BP
        self.x_l = []     # 2d array of the x at each step

        # parameters for loss function and activation
        self.activation = "todo"    # activation function
        self.loss = "todo"          # loss function
        
        # parameters describing shape of the neural net
        self.n_inputs = inputs       # no. inputs (excludes x0 = 1)
        self.n_outputs = outputs     # no. output heads
        self.h_layers = h_layers     # no. hidden layers
        self.h_layers_d = h_layers_d # no. nodes/layer (excluding zeroth coor)
        self.total_layers = h_layers + 1  # index of last layer
        
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
        print("Strela Net initialized:")
        print("%d inputs, %d outputs" %(inputs, outputs))
        print("%d hidden layers, %d nodes per hidden layer"\
         %(h_layers, h_layers_d))
                    
        return
    
    def train(self, x_all, y_all, passes = 25):
        '''trains the neural net using stochastic gradient descent
        inputs: x_all - 2d list like object of all x inputs 
                y_all - 2d array of correct classifications
                passes - number of passes through data
        outputs: none'''

        x_all = np.array(x_all)        # convert x and y to numpy arrays
        y_all = np.array(y_all)

        # total number of points to train on
        total_train = len(x_all) * passes
        # number of points trained on so far
        current_train = 0
        
        for i in range(passes):
            # perform multiple passes through the data
            order = np.arange(len(x_all))
            # array of indexes in random order - used for SGD   
            np.random.shuffle(order)
            # train on the points in a random order
            for n in order:
                # generate the prediction; pass 2D array and get back 2D array;
                # pull the zeroth element of return to have 1D array prediction
                y_pre = self.predict([x_all[n]])[0]
                # calculate the sigma of the last layer - layer L
                # wrap in brackets and convert to np array to make 2D array
                self.delta_l[self.total_layers] = np.array([2 * (1 - y_pre**2) * (y_pre - y_all[n])])

                # go backwards in L and calculate the deltas
                for l in reversed(range(2, self.total_layers + 1)):
                    # back-propagate the deltas
                    self.delta_l[l-1] = (1 - np.array(self.x_l[l-1])**2) * \
                                        np.dot(self.weights[l], self.delta_l[l])
                # now update the weights
                for l in range(1, self.total_layers + 1):
                    # update weights of each layer
                    self.weights[l] -=\
                        self.lr * np.dot(self.x_l[l-1], np.transpose(self.delta_l[l]))
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
            self.x_l[0][1:] = np.array(x_all[n]).reshape(self.n_inputs, 1)
            for l in range(1, self.total_layers + 1):            
                # mutliply weights by x of previous layer to get intermediate s
                s_j = np.transpose(np.dot(np.transpose(self.x_l[l-1]), self.weights[l]))
                # apply activation to get x_l               
                self.x_l[l] = self.activation(s_j)
            # prediction is the output of the final layer
            # reshape to be one dimensional array
            y_pre[n] = (self.x_l[self.total_layers]).reshape(self.n_outputs,)
        # return array of predictions
        return y_pre

    def activation(self, s):
        '''activation function; currently hard-coded as tanh. Recursively 
         applies the activation function to a np object (array or float)
         of any shape elementwise. The return value should have the same
         dimensions as the argument
         inputs: s - object to apply activation function to
         outputs: theta - new object with activation applied'''
        theta = []
        # base case - check it's a float/int
        if np.shape(s) == ():
            return(np.exp(s) - np.exp(-s))/(np.exp(s) + np.exp(-s))
        else:
             # otherwise iterate through and perform recursive call
            for x in s:
                theta.append(self.activation(x))
        return np.array(theta)

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



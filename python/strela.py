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

import numpy as np

class strela_net():
    '''this class is a neural net that can be used for classifcation and
       prediction. The class does not inherit from a superclass'''
    def __init__(self, inputs, outputs, h_layers, h_layers_d, lr = 0.01):
        '''initializes the neural net. Creates a matrix for the weights
        and randomly initializes them
        inputs: inputs - number of inputs to the net
                outputs - number of output heads to the net
                h_layers - number of hidden layers
                h_layers_d - dimension of each hidden layer
        outputs: none'''
        
        # learning parameters
        self.lr = lr      # learning rate        
        
        # parameters that change during training
        self.weights = [] # current weights
        self.delta_l = [] # 2d array of delta from most recent BP
        self.x_l = []     # 2d array of the x at each step
        
        # parameters describing shape of the neural net
        self.n_inputs = inputs       # no. inputs (excludes x0 = 1)
        self.n_outputs = outputs     # no. output heads
        self.h_layers = h_layers     # no. hidden layers
        self.h_layers_d = h_layers_d # no. nodes/layer (excluding zeroth coor)
        self.total_layers = h_layers + 1
        
        # arrays describing shape of the neural net
        # total number of layers (L) is no. hidden layers + 1 input layer + 
        # 1 output layer; weight indexing starts at 1 so the 0th layer
        # is unoccupied, necessitating adding 3 rather than 2
        
        # inputs to each layer - init all to dimension of hidden layers
        self.i_l = [self.h_layers_d for i in range(self.total_layers + 1)] 
        # outputs to each layer - init all to dimension of hidden layers
        self.j_l = [self.h_layers_d for j in range(self.total_layers + 1)] 
        
        # fill in the special cases
        self.i_l[0] = self.n_inputs      # layer indexing starts at 1
        self.j_l[0] = 0
        self.j_l[self.total_layers] = self.n_outputs    # last output layer takes n_outputs

        
        # set up the weights, delta, and x arrays to be the correct shape
        for l in range(self.total_layers + 1):
            # randomly initialize the weights
            self.weights.append\
                (0.01 * (np.random.rand(self.i_l[l-1] + 1, self.j_l[l] + 1) - 0.5))
            # + 1 term is for the input 1 term at each layer
            
            # set up the dimensions of the x outputs
            if l == 0:
                # "output" of zeroeth layer is the x point being trained on
                self.x_l.append(np.zeros((self.n_inputs + 1, 1)))
            elif l == self.total_layers:
                self.x_l.append(np.zeros((self.n_outputs + 1, 1)))
            else:
                self.x_l.append(np.zeros((self.h_layers_d + 1, 1)))
                
            # set up dimensions of delta
            self.delta_l.append(np.zeros((self.j_l[l] + 1, 1)))

            print("\nlayer:", l)
            print("delta: ", np.shape(self.delta_l[l]))
            print("weights ", np.shape(self.weights[l]))

            
        # set up the xl bias terms
        for l in range(self.total_layers):
            self.x_l[l][0][0] = 1
                    
        return
    

    
    def train(self, x_all, y_all):
        '''trains the neural net using stochastic gradient descent
        inputs: x_all - 2d list like object of all x inputs 
                y_all - 2d array of correct classifications
        outputs: none'''
        
        order = np.arange(len(x_all))
        # array is an array of indexes in random order    
        np.random.shuffle(order)
        # train on the points in a random order    
        for n in order:
            # generate the prediction; pass 2D array and get back 2D array;
            # pull the zeroth element of return to have 1D array prediction
            print("\n\nTest point: ", n, "  ", x_all[n], y_all[n])
            print("initial x_l:")
            print(self.x_l)
            y_pre = self.predict([x_all[n]])[0]
            print("updated x_l:")
            print(self.x_l)
            # calculate the sigma of the last layer - layer L
            print("Before Predicted: ", y_pre, "  Actual: ", y_all[n])
            print("Squared error: ", (y_pre - y_all[n])**2)
            # check this step
            final_delta = np.array([2 * (1 - y_pre**2) * (y_pre - y_all[n])])
            self.delta_l[self.total_layers] = np.concatenate((np.array([[0]]), final_delta), axis = 0)
            print(self.delta_l[self.total_layers])

            # go backwards in L and calculate the deltas
            for l in reversed(range(2, self.total_layers + 1)):
                # back-propagate the deltas
                self.delta_l[l-1][1:] = (1 - np.array(self.x_l[l-1])**2)[1:] * \
                                    np.dot(self.weights[l], self.delta_l[l])[1:]
            print("Delta:")
            print(self.delta_l)
            # now update the weights
            print("previous weights:")
            self.show_weights()
            for l in range(1, self.total_layers + 1):
                # update weights of each layer
                self.weights[l] -=\
                    self.lr * np.dot(self.x_l[l-1], np.transpose(self.delta_l[l]))
            print("updated weights:")
            self.show_weights()
            y_pre = self.predict([x_all[n]])[0]
            print("After: ", y_pre)
            print("Squared error: ", (y_pre - y_all[n])**2)

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
                
                # special case - final layer x_l is simply post activation
                # should be able to remove this special case
                #if l == self.total_layers:
                #    self.x_l[l] = self.activation(s_j)[1:]
                # all other layers - don't overwrite 1
                #else:
                    # apply the activation and get the result
                self.x_l[l][1:] = self.activation(s_j)[1:]
                # print("layer index: ", l)
            # prediction is the output of the final layer
            # print(self.x_l[self.total_layers])
            # reshape to be one dimensional array; zeroeth term is not applicable
            print(self.x_l)
            y_pre[n] = (self.x_l[self.total_layers]).reshape(self.n_outputs + 1,)[1:]

        
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
        
    def show_weights(self):
        '''prints the weights of the neural net in pretty form'''
        for l in range(self.total_layers + 1):
            print("Layer: ", l)
            print(self.weights[l])
        
        return
    
    def save_weights(self):
        '''saves the weights to a text file and a pickle file'''
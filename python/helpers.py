# helpers test playground
#
# this file is for testing individual helper functions in the strela
# class. Functions in this file may then be directly copied with small
# modifications to the strela file
#
# Revision History
#


import numpy as np

def add(s):
    '''add recursively adds 1 to every element in a nested array; used to
    test the recursive function activiation. The return should preserve
    the shape of the passed array'''
    

    result = []
    
    if type(s) == type(8):
        return(s + 1)
    else:
        for x in s:
            result.append(add(x))
    
    return result


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
    for l in range(1, self.total_layers + 1):
	print("Layer: ", l)
	print(self.weights[l])
    
    return

def save_weights(self):
    '''saves the weights to a text file and a pickle file'''


'''activation function; currently hard-coded as tanh. Recursively 
 applies the activation function to a np object (array or float)
 of any shape elementwise. The return value should have the same
 dimensions as the argument
 inputs: s - object to apply activation function to
 outputs: theta - new object with activation applied'''
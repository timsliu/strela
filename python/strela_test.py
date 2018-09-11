# strela neural net testing program
#
# this file contains functions that test the strela neural net
#
# Table of Contents
#
# Revision History
# 07/17/18    Tim Liu    started file and wrote test_predict
# 09/10/18    Tim Liu    wrote test square to set weights 
# 09/10/18    Tim Liu    modified train_plot to test a circle
#                        in two dimensions


from strela import strela_net
import numpy as np
# allow for reloading without closing terminal
import imp
import strela_test
import matplotlib.pyplot as plt

def test_predict(n_in, n_out, h_layers = 3, h_layers_d = 5):
    '''simple test for the predict method of strela_net. Creates an instance
    of strela net and randomly creates inputs to generate predictions on.
    Function tests that the neural net correctly calculates the output;
    does not test the training or back propagation
    inputs: n_in - (int) dimensions of each input point
            n_out - (int) dimensionality of the predicted output
            h_layers - number of hidden layers
            h_layers_d = nodes per hidden layer'''
    
    print("Creating new strela net")
    print("Hidden layers: %d  Nodes per layer: %d" %(h_layers, h_layers_d))
    print("Input dim:     %d  Output dim:      %d" %(n_in, n_out))
    # new instance of strela net
    my_strela = strela_net(n_in, n_out, h_layers, h_layers_d)
    # generate 10 random data points of dimension n_in
    x_in = np.random.rand(10, n_in)
    y_pre = my_strela.predict(x_in)
    print("Predicted values:")
    print(y_pre)
    return

def test_train(h_layers = 1, h_layers_d = 5, lr = 0.01):
    '''stimple test of strela net training. Randomly generates points in a 
    multi-dimensional space and trains the net on them. Then evaluates on
    a separate test set. The training set has multiple x inputs and
    a single y
    
    This function tests a linearly separable dataset with a single y
    coordinate'''
    
    n_input = 2         # dimensionality of the space
    train_size = 500   # size of the training set
    test_size = 100     # size of the test set
    
    # generate random x values
    x_train = 10 * (np.random.rand(train_size, n_input) - 0.5)
    x_test = 10 * (np.random.rand(test_size, n_input) - 0.5)    
    # generate y values 
    y_train = [1 if x[0] + x[1] ** 2 > 2 else -1 for x in x_train]
    y_test = [1 if x[0] + x[1] ** 2 > 2 else -1 for x in x_test]
    
    print("y_test:")
    print(y_test)
    
    # create instance of strela net
    my_strela = strela_net(n_input, 1, h_layers, h_layers_d, lr)
    # train the net
    my_strela.train(x_train, y_train)
    
    # predict the test set
    print("Generating predictions on test set...")
    y_pre_raw = my_strela.predict(x_test)
    #print(y_pre_raw)
    # apply floor function to test set
    print("Applying floor function...")
    y_pre = [1 if x[0] > 0 else -1 for x in y_pre_raw]
    
    # check if predictions match actual values
    correct = 0
    for i in range(test_size):
        if y_pre[i] == y_test[i]:
            correct += 1
            
    print("Fraction correctly classified: ", correct/test_size)


    return 


def train_plot(h_layers = 1, h_layers_d = 5, lr = 0.01):
    '''plots the classification done by the neural net along with the
    actual function in two dimensions'''
    
    n_input = 2         # dimensionality of the space
    train_size = 1000   # size of the training set
    test_size = 1000     # size of the test set
    
    # generate random x values
    x_train = 10 * (np.random.rand(train_size, n_input) - 0.5)
    x_test = 10 * (np.random.rand(test_size, n_input) - 0.5)    
    # generate y values 
    y_train = [1 if (x[1] + 1)**2 + x[0]**2 > 9 else -1 for x in x_train]
    y_test = [1 if (x[1] + 1)**2 + x[0]**2 > 9 else -1 for x in x_test]
    
    print("y_test:")
    print(y_test)
    
    # create instance of strela net
    my_strela = strela_net(n_input, 1, h_layers, h_layers_d, lr)
    # train the net
    my_strela.train(x_train, y_train)
    
    # predict the test set
    print("Generating predictions on test set...")
    y_pre_raw = my_strela.predict(x_test)
    # apply floor function to test set
    print("Applying floor function...")
    y_pre = [1 if x[0] > 0 else -1 for x in y_pre_raw]
    
    # check if predictions match actual values
    correct = 0
    for i in range(test_size):
        if y_pre[i] == y_test[i]:
            correct += 1
            
    print("Fraction correctly classified: ", correct/test_size)  
    print("Number correctly classified: ", correct)  


    c = ['red' if x == 1 else 'blue' for x in y_pre]
    x_line = np.arange(-3, 3, 0.01)
    y_line = [(9-x**2) ** 0.5 - 1 for x in x_line]
    y_line2 = [-(9-x**2) ** 0.5 - 1 for x in x_line]

    plt.scatter([x[0] for x in x_test], [x[1] for x in x_test], color = c) 
    plt.plot(x_line, y_line, color = 'green')
    plt.plot(x_line, y_line2, color = 'green')

    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
    plt.show() 
    
    my_strela.show_weights()
    return 

def test_square():
    my_strela = strela_net(2, 1, 0, 0)

    '''test to see if neural net is able to generate non-linear predictions
    Uses the set weights function to set the weights and then runs predictions
    and plots the result'''

    # set up a neural net with 2 inputs 1 output and no hidden layer
    my_strela.set_weights(1, [[-1.5], [1], [1]])

    x_train = []

    n_input = 2         # dimensionality of the space
    train_size = 1000   # size of the training set
    # generate random x values
    x_train = 5 * (np.random.rand(train_size, n_input))
    # generate y values 
    y_pre_raw = my_strela.predict(x_train)
    # apply floor function to test set
    print("Applying floor function...")
    y_pre = [1 if x[0] > 0 else -1 for x in y_pre_raw]
    c = ['red' if x == 1 else 'blue' for x in y_pre]

    plt.scatter([x[0] for x in x_train], [x[1] for x in x_train], color = c)
    plt.show()
    return


def test_simple(lr = 0.1):
    '''hard coded version of test_train. Instead of a randomly generated
    test set, this function tests the train method on a linearly separable
    one dimensional test set. The neural net is initialized to have
    no hidden layer

    inputs: lr - learning rate'''

    test_size = 100
    
    
    # generate random x values
    x_train = [[-0.5], [0.7], [-0.9], [0.3], [0.8], [-0.2], [0.8], [-0.6], [-0.6]]
    x_test = 10 * (np.random.rand(test_size, 1) - 0.5)    
    # generate y values; separating line is at 0
    y_train = [1 if sum(x) > 0 else -1 for x in x_train]
    y_test = [1 if sum(x) > 0 else -1 for x in x_test]
    
    print("y_test:")
    print(y_test)
    
    # create instance of strela net
    # 1 input 1 output no hidden layer
    my_strela = strela_net(1, 1, 1, 5, lr)
    # train the net
    my_strela.train(x_train, y_train)
    
    # predict the test set
    y_pre_raw = my_strela.predict(x_test)
    print("Predictions:")
    print(y_pre_raw)
    # apply floor function to test set
    y_pre = [1 if x[0] > 0 else -1 for x in y_pre_raw]
    print("y_predicted:")
    print(y_pre)
    
    # check if predictions match actual values
    correct = 0
    for i in range(test_size):
        if y_pre[i] == y_test[i]:
            correct += 1
    print("Predictions:")
    print(y_pre)
            
    print("Fraction correctly classified: ", correct/test_size)
    
    return
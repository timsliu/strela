# strela neural net testing program
#
# this file contains functions that test the strela neural net
#
# Table of Contents
#
# Revision History
# 07/17/18    Tim Liu    started file and wrote test_predict
#


from strela import strela_net
import numpy as np
# allow for reloading without closing terminal
import imp
import strela_test

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

def test_train(h_layers = 2, h_layers_d = 2, lr = 0.1):
    '''stimple test of strela net training. Randomly generates points in a 
    multi-dimensional space and trains the net on them. Then evaluates on
    a separate test set. The training set has multiple x inputs and
    a single y
    
    This function tests a linearly separable dataset with a single y
    coordinate'''
    
    n_input = 5         # dimensionality of the space
    train_size = 100   # size of the training set
    test_size = 30     # size of the test set
    
    # generate random x values
    x_train = 10 * (np.random.rand(train_size, n_input) - 0.5)
    x_test = 10 * (np.random.rand(test_size, n_input) - 0.5)    
    # generate y values; hyperplane separating points is sum of coefficients
    # equalling 1
    y_train = [1 if sum(x) > 1 else -1 for x in x_train]
    y_test = [1 if sum(x) > 1 else -1 for x in x_test]
    
    print("y_test:")
    print(y_test)
    
    # create instance of strela net
    my_strela = strela_net(n_input, 1, h_layers, h_layers_d, lr)
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
            
    print("Fraction correctly classified: ", correct/test_size)
    
    #my_strela.show_weights()
    #print(my_strela.x_l)
    
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
    my_strela = strela_net(1, 1, 0, 0, lr)
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
            
    print("Fraction correctly classified: ", correct/test_size)
    
    return
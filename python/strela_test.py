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
# 09/17/18    Tim Liu    removed test_square and updated comments
# 09/28/18    Tim Liu    wrote test_erf for testing different error functions
# 09/30/18    Tim Liu    removed output encoding from test functions;
#                        this feature now done by strela_net


from strela import *
import numpy as np
import matplotlib.pyplot as plt

def test_predict(n_in, n_out, h_layers = 3, h_layers_d = 5):
    '''simple test for the predict method of strela_net. Creates an instance
    of strela net and randomly creates inputs to generate predictions on.
    Function tests that the neural net calculates the output and that arrays
    are aligned;
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
    '''simple test of strela net training. Randomly generates points in a 
    multi-dimensional space and trains the net on them. Then evaluates on
    a separate test set. The training set has multiple x inputs and
    a single y'''
    
    n_input = 4         # dimensionality of the space
    train_size = 500   # size of the training set
    test_size = 100     # size of the test set
    
    # generate random x values
    x_train = 10 * (np.random.rand(train_size, n_input) - 0.5)
    x_test = 10 * (np.random.rand(test_size, n_input) - 0.5)    
    # generate y values - change function here
    y_train = [1 if x[0] + x[1] ** 2 + x[2] ** 3 > 2 else -1 for x in x_train]
    y_test = [1 if x[0] + x[1] ** 2 + x[2] ** 3 > 2 else -1 for x in x_test]
    
    
    # create instance of strela net
    my_strela = strela_net(n_input, 1, h_layers, h_layers_d, lr, out_encode = "binary")
    # train the net
    my_strela.train(x_train, y_train)
    
    # predict the test set
    print("Generating predictions on test set...")
    y_pre = my_strela.predict(x_test)
    
    # check if predictions match actual values
    correct = 0
    for i in range(test_size):
        if y_pre[i] == y_test[i]:
            correct += 1
            
    print("Fraction correctly classified: ", correct/test_size)


    return

def test_multiclass(h_layers = 1, h_layers_d = 40, lr = 1e-3):
    '''test of multiclass classification.
    inputs: h_layers - number of hidden layers
            h_layers_d = number of nodes per hidden layer
            lr - learning rate'''
    
    n_input = 2         # dimensionality of the space
    n_output = 3        # number of classes
    train_size = 500    # size of the training set
    test_size = 500     # size of the test set
    
    # generate random x values
    x_train = 10 * (np.random.rand(train_size, n_input) - 0.5)
    x_test = 10 * (np.random.rand(test_size, n_input) - 0.5)    
    # use helper function to create y_values
    y_train = tag_multiclass(x_train, n_output)
    y_test = tag_multiclass(x_test, n_output)
    
    
    # create instance of strela net; apply softmax and use categorical cross entropy
    my_strela = strela_net(n_input, n_output, h_layers, h_layers_d, lr,\
     epochs = 25, softmax = True, loss = "squared", out_encode = "one_hot")
    # train the net
    my_strela.train(x_train, y_train)
    
    # predict the test set
    print("Generating predictions on test set...")
    y_pre = my_strela.predict(x_test)

    # check if predictions match actual values
    correct = 0
    for i in range(test_size):
        if np.array_equal(y_pre[i], y_test[i]):
            correct += 1
            
    print("Fraction correctly classified: ", correct/test_size)

    # color the predictions
    color_dic = {0: "red", 1: "green", 2: "blue"}
    c = [color_dic[list(x).index(1)] for x in y_pre]
    # draw separating line
    x_0 = np.arange(0, 5, 0.01)
    y_0 = [(2-x)/2 for x in x_0]

    x_1 = [0, 0]
    y_1 = [-5, 1]

    x_2 = np.arange(-5, 0, 0.01)
    y_2 = [x**2 - 1 for x in x_2]

    x_3 = np.arange(1.2, 5, 0.01)
    y_3 = [x**2 - 1 for x in x_3]

    # plot the points and the separating line
    plt.scatter([x[0] for x in x_test], [x[1] for x in x_test], color = c) 
    plt.plot(x_0, y_0, color = 'black')
    plt.plot(x_1, y_1, color = 'black')
    plt.plot(x_2, y_2, color = 'black')
    plt.plot(x_3, y_3, color = 'black')


    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
    plt.show() 

    return

def tag_multiclass(x_all, classes):
    '''helper function for tagging a multiclass dataset
    inputs: x_all - x array to tag
            classes - number of categories
    outputs: y_all - one hot encoded tags'''

    # array of one hot encoded y classifications
    y_all = np.zeros((len(x_all), classes))
    # list of the indices to set to 1
    tags = []

    for x in x_all:
        # classify the points
        if (x[0] + 2 * x[1] < 2) and (x[0] > 0):
            # group 0
            tags.append(0)
        elif x[0] ** 2 - x[1] > 1:
            # group 1
            tags.append(1)
        else:
            tags.append(2)

    for t in range(len(tags)):
        # create one hot encoding
        y_all[t][tags[t]] = 1

    return y_all

def train_plot(h_layers = 1, h_layers_d = 5, lr = 0.01):
    '''plots the classification done by the neural net along with the
    actual function in two dimensions. Variation of test_train with
    added plotting function.
    inputs: h_layers - number of hidden layers
            h_layers_d - number of nodes per hidden layer
            lr - learning rate'''
    
    n_input = 2          # dimensionality of the space
    train_size = 1000    # size of the training set
    test_size = 1000     # size of the test set
    
    # generate random x values
    x_train = 10 * (np.random.rand(train_size, n_input) - 0.5)
    x_test = 10 * (np.random.rand(test_size, n_input) - 0.5)    
    # generate y values 
    y_train = [1 if (x[1] + 1)**2 + x[0]**2 > 9 else -1 for x in x_train]
    y_test = [1 if (x[1] + 1)**2 + x[0]**2 > 9 else -1 for x in x_test]
    

    
    # create instance of strela net
    my_strela = strela_net(n_input, 1, h_layers, h_layers_d, lr, out_encode = "binary")
    # train the net
    my_strela.train(x_train, y_train)
    
    # predict the test set
    print("Generating predictions on test set...")
    y_pre = my_strela.predict(x_test)
    
    # check if predictions match actual values
    correct = 0
    for i in range(test_size):
        if y_pre[i] == y_test[i]:
            correct += 1
            
    print("Fraction correctly classified: ", correct/test_size)  
    print("Number correctly classified: ", correct)  


    # color the predictions
    c = ['red' if x == 1 else 'blue' for x in y_pre]
    # draw separating line
    x_line = np.arange(-3, 3, 0.01)
    y_line = [(9-x**2) ** 0.5 - 1 for x in x_line]
    y_line2 = [-(9-x**2) ** 0.5 - 1 for x in x_line]

    # plot the points and the separating line
    plt.scatter([x[0] for x in x_test], [x[1] for x in x_test], color = c) 
    plt.plot(x_line, y_line, color = 'green')
    plt.plot(x_line, y_line2, color = 'green')

    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
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
    
    
    # create instance of strela net
    # 1 input 1 output no hidden layer
    my_strela = strela_net(1, 1, 1, 5, lr, out_encode = "binary")
    # train the net
    my_strela.train(x_train, y_train)
    
    # predict the test set
    y_pre = my_strela.predict(x_test)
    # check if predictions match actual values
    correct = 0
    for i in range(test_size):
        if y_pre[i] == y_test[i]:
            correct += 1
    # print("Predictions:")
    # print(y_pre)
            
    print("Fraction correctly classified: ", correct/test_size)
    
    return

def test_reg(h_layers = 2, h_layers_d = 10, lr = 0.01):
    '''test of regularization. The training set and test set
    are drawn from the same function but the training set has
    incorrectly labeled points thrown in'''

    n_input = 2          # dimensionality of the space
    train_size = 1000    # size of the training set
    test_size = 1000     # size of the test set
    
    # generate random x values
    x_train = 10 * (np.random.rand(train_size, n_input) - 0.5)
    x_test = 10 * (np.random.rand(test_size, n_input) - 0.5)    
    # generate y values 
    y_train = [1 if x[1] + x[0]**2 > 9 else -1 for x in x_train]
    y_test = [1 if x[1] + x[0]**2 > 9 else -1 for x in x_test]
    # now randomly flip some of the training set values
    flip = np.random.random_integers(0, train_size-1, round(train_size/10))

    for i in flip:
        # flip some of the training set labels; some may get flipped
        # more than once
        y_train[i] *= -1

    regs = [0, 1e-8, 1e-6, 1e-4, 1e-2, 1]   # list of regularization levels to test
    in_sample_error = []
    out_sample_error = []

    for l in regs:
        # create instance of strela net
        my_strela = strela_net(n_input, 1, h_layers, h_layers_d, lr, reg = l, out_encode = "binary", epochs = 10)
        # train the net
        my_strela.train(x_train, y_train)

        # in sample performance - predict the training set
        y_pre = my_strela.predict(x_train)  
        # check if predictions match actual values
        correct = 0
        for i in range(train_size):
            if y_pre[i] == y_train[i]:
                correct += 1
        in_sample_error.append(correct/train_size)


        # out of sample performance - predict the test set
        y_pre = my_strela.predict(x_test)   
        # check if predictions match actual values
        correct = 0
        for i in range(test_size):
            if y_pre[i] == y_test[i]:
                correct += 1
        out_sample_error.append(correct/test_size)

    print("Regularizations: ", regs)
    print("In sample: ", in_sample_error)
    print("Out sample: ", out_sample_error)

    return






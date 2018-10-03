# helpers for strela
#
# This file contains helper functions for strela
#
# Revision History
# 09/19/18    Tim Liu    created file
# 09/27/18    Tim Liu    added convert_one_hot and convert_binary
# 09/30/18    Tim Liu    updated convert_one_hot and convert_binary
#                        to work on 1D (n,) arrays
# 10/01/18    Tim Liu    updated documentation
#
# Table of contents
#
# make_2d - reshape array into a numpy 2D array
# apply_softmax - apply softmax function to an array
# convert_one_hot - applies one hot encoding to a 1D array
# convert_binary - applies binary encoding to a 1D array


import numpy as np

def make_2d(in_array):
	'''turns an array like object into a numpy 2D array. If the obeject
	is already a 2D array it is unchanged. If it is a 1D array it is changed
	to a 2D (n, 1) array'''

	out_array = np.array(in_array)

    # reject arrays with too many dimensions
	if len(np.shape(out_array)) > 2:
		print("make_2d only reshapes 2d or 1d arrays")
		return

	# reshape 1D arrays into 2D
	if len(np.shape(out_array)) == 1:
		length = len(out_array)
		out_array = np.reshape(out_array, (length, 1))

	return out_array

def apply_softmax(in_array):
	'''returns the softmax function for an array
	inputs: in_array - 1D array
	outputs: softmax - array with softmax applied'''

    # convert to numpy array 
	in_array = np.array(in_array)
	# softmax function
	softmax = np.exp(in_array)/np.sum(np.exp(in_array))

	return softmax

def convert_one_hot(in_array):
	'''takes a 1d (n,) array and converts to one hot encoding. The
	largest value is set to 1 and all other values are set to 0.
	Used for multiclass classification'''

	largest_index = 0
	largest = -np.inf

    # search for largest value
	for i in range(len(in_array)):
		if in_array[i] > largest:
			largest_index = i
			largest = in_array[i]

    # create array of zeros
	one_hot = np.zeros(len(in_array))
    # one hot encode it
	one_hot[largest_index] = 1

	return one_hot

def convert_binary(in_array):
	'''applies step function to 1D (n,) array; values greater than or equal to
	 0 are set to 1 and values less than 0 are set to -1'''
	return [-1 if x < 0 else 1 for x in in_array]



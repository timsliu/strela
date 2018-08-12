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
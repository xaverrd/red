from numpy import exp, vectorize, array
from numpy.random import permutation

def sigmoid(x):
    return 1/(1+exp(-x))
sigmoid = vectorize(sigmoid)

def sigmoid_slope(x):
    return sigmoid(x)*(1-sigmoid(x))
sigmoid_slope = vectorize(sigmoid_slope)

def squared_difference(x, y):
    return (x-y)**2
squared_difference = vectorize(squared_difference)

def squared_difference_slope(x, y):
    return 2*(x-y)
squared_difference_slope = vectorize(squared_difference_slope)

def shuffle(x, y):
    randomize = permutation(len(x)).tolist()
    return array(x)[randomize].tolist(), array(y)[randomize].tolist()

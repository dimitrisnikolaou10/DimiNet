import numpy as np


def sigmoid_function(W, X):
    return 1 / (1 + np.exp(-np.dot(X, W)))


def standard_product(W, X):
    return np.dot(X, W)
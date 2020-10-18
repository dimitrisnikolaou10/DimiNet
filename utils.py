import numpy as np


def __sigmoid_function(self, W, X):
    return 1 / (1 + np.exp(-np.dot(W, X)))
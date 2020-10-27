import unittest
import numpy as np
from src.utils import sigmoid_function  # if directory unique no need for relative


class TestSigmoid(unittest.TestCase):
    def test_sigmoid(self):
        W = np.ones((10, 1))
        X = np.ones((5, 10))
        result = sigmoid_function(W, X)
        sum_of_result = sum(result)[0]
        correct = 4.999773010656488
        self.assertEqual(sum_of_result, correct, "Sigmoid Function is not working properly.")


if __name__ == '__main__':
    unittest.main()




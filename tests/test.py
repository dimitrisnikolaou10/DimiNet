import unittest
import numpy as np
import pandas as pd
from src.utils import sigmoid_function  # if directory unique no need for relative
from src.Model import LinearRegression, LogisticRegression
from src.Metrics import mean_absolute_error, accuracy


class TestSigmoid(unittest.TestCase):
    def test_sigmoid(self):
        W = np.ones((10, 1))
        X = np.ones((5, 10))
        result = sigmoid_function(W, X)
        sum_of_result = sum(result)[0]
        correct = 4.999773010656488
        self.assertEqual(sum_of_result, correct, "Sigmoid Function is not working properly.")


class TestGradientDescent(unittest.TestCase):
    def test_gradient_descent(self):

        df = pd.read_csv("../lib/Fish.csv")
        df = pd.concat((df, pd.get_dummies(df.Species)), 1)
        y = df["Weight"].values.reshape((df["Weight"].values.shape[0], 1))
        del df["Species"], df["Weight"]

        X = df.values
        lr = LinearRegression()
        loss_curve = lr.fit(X, y, epochs=np.power(10, 3), print_loss_every=10000000)
        y_pred = lr.predict(X)

        errors = []
        for i, y in zip(y, y_pred):
            errors.append(i[0]-y[0])

        res = mean_absolute_error(errors)
        correct = 238.6563103230453
        self.assertEqual(res, correct, "Gradient descent is not working properly.")


class TestLogisticRegression(unittest.TestCase):
    def test_logistic_regression(self):

        df = pd.read_csv("../lib/titanic.csv")
        del df["Unnamed: 0"]
        df.dropna(how="any", inplace=True)
        y = df["Survived"].values.reshape((df["Survived"].values.shape[0], 1))
        del df["Survived"]

        X = df.values
        lr = LogisticRegression(regularize="RidgeClassification")
        loss_curve = lr.fit(X, y, epochs=np.power(10, 3), print_loss_every=1000000000)
        y_pred = lr.predict(X)

        res = accuracy(y, y_pred)
        correct = 0.5938375350140056
        self.assertEqual(res, correct, "Logistic Regression is not working properly.")




if __name__ == '__main__':
    unittest.main()




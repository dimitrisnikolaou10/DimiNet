from src.Optimizer import GradientDescent
import numpy as np
from src import utils


class LinearRegression:
    """
    Class for the Linear Regression model.
    Methods:
        - fit:
            Runs the optimization
            - X: training data (numpy array)
            - y labels for training data (numpy array)
            - optimization: algorithm used to optimize
            - lambda_1: Lasso regularization parameter
            - lambda_2: Ridge regularization parameter
        - predict:
            Generate predictions based on the trained weights.
            - X: testing data (numpy array)
    Constuctor args:
        - regularize: Type of regularization (string)
        - normalize: If we should normalize the data (boolean)
        - intercept: If there should be an intercept (boolean)
    """
    def __init__(self, regularize=None, normalize=True, intercept=True):
        self.normalize=normalize
        self.trained = False
        self.intercept = intercept
        if not regularize:
            self.model_name = "LinearRegression"
        elif regularize not in ["LassoRegression", "RidgeRegression"]:
            raise ValueError("Regularization method must take either the value Lasso or Ridge.")
        else:
            self.model_name=regularize
        self.W = None

    def fit(self, X, y, optimization="GradientDescent", lambda_1=None, lambda_2=None):
        if self.normalize:
            Xmax, Xmin = X.max(), X.min()
            X = (X - Xmin)/(Xmax - Xmin)
        if self.intercept:
            X = np.append(X, np.ones((X.shape[0], 1)), axis=1)

        self.W = np.zeros((X.shape[1], 1))

        if self.model_name=="LassoRegression" and lambda_1 is None:
            raise ValueError("Lasso regression required a regularization parameter lambda_1")
        elif self.model_name=="RidgeRegression" and lambda_2 is None:
            raise ValueError("Ridge regression required a regularization parameter lambda 2")


        if optimization == "GradientDescent":
            if self.model_name=="LassoRegression":
                raise Exception("Lasso loss function is not differentiable.")
            optimizer = GradientDescent(X, y, self.model_name, W=self.W)
            self.W, epoch_losses = optimizer.optimize(lambda_1 = lambda_1, lambda_2 = lambda_2, batch_size=30)
            self.trained = True
        elif optimization == "ClosedForm":
            if self.model_name=="LinearRegression":
                self.W = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
            elif self.model_name=="LassoRegression":
                raise Exception("Lasso does not generalise to a closed form solution.")
            elif self.model_name=="RidgeRegression":
                self.W = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + lambda_2*np.eye(X.shape[0])), X.T), y)
            epoch_losses = None
            self.trained = True

        return epoch_losses

    def predict(self, X):
        if not self.trained:
            raise Exception("Please train your model before generating predictions.")
        if self.normalize:
            Xmax, Xmin = X.max(), X.min()
            X = (X - Xmin)/(Xmax - Xmin)
        if self.intercept:
            X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        y_pred = np.dot(X, self.W)
        return y_pred


class LogisticRegression:
    """
    Class for the Logistic Regression model.
    Methods:
        - fit:
            Runs the optimization
            - X: training data (numpy array)
            - y: labels for training data (numpy array)
            - optimization: algorithm used to optimize
            - lambda_1: Lasso regularization parameter
            - lambda_2: Ridge regularization parameter
        - predict:
            Generate predictions based on the trained weights.
            - X: testing data (numpy array)
    Constuctor args:
        - regularize: Type of regularization (string)
        - normalize: If we should normalize the data (boolean)
        - intercept: If there should be an intercept (boolean)
    """
    def __init__(self, regularize=None, normalize=True, intercept=True):
        self.normalize = normalize
        self.trained = False
        self.intercept = intercept
        if not regularize:
            self.model_name = "LogisticRegression"
        elif regularize not in ["LassoClassification", "RidgeClassification"]:
            raise ValueError("Regularization method must take either the value Lasso or Ridge.")
        else:
            self.model_name = regularize
        self.W = None

    def fit(self, X, y, optimization="GradientDescent", lambda_1=0.001, lambda_2=0.001):
        if self.normalize:
            Xmax, Xmin = X.max(), X.min()
            X = (X - Xmin)/(Xmax - Xmin)
        if self.intercept:
            X = np.append(X, np.ones((X.shape[0], 1)), axis=1)

        self.W = np.zeros((X.shape[1], 1))

        if self.model_name=="LassoClassification" and lambda_1 is None:
            raise ValueError("Lasso classification required a regularization parameter lambda_1")
        elif self.model_name=="RidgeClassification" and lambda_2 is None:
            raise ValueError("Ridge classification required a regularization parameter lambda 2")


        if optimization == "GradientDescent":
            optimizer = GradientDescent(X, y, self.model_name, W=self.W)
            self.W, epoch_losses = optimizer.optimize(lambda_2=lambda_2, batch_size=30)
            self.trained = True

        return epoch_losses

    def predict(self, X):
        if not self.trained:
            raise Exception("Please train your model before generating predictions.")
        if self.normalize:
            Xmax, Xmin = X.max(), X.min()
            X = (X - Xmin)/(Xmax - Xmin)
        if self.intercept:
            X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        y_pred = utils.sigmoid_function(self.W, X)
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
        return np.squeeze(y_pred)

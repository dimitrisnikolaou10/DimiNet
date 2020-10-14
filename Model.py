from Optimizer import GradientDescent
import numpy as np


class LinearRegression:
    def __init__(self, regularize=None, normalize=True, intercept=True):
        self.normalize=normalize
        self.trained = False
        self.intercept = intercept
        if not regularize:
            self.model_name="LinearRegression"
        elif regularize not in ["Lasso", "Ridge"]:
            raise ValueError("Regularization method must take either the value Lasso or Ridge.")
        else:
            self.model_name=regularize
        self.W = None

    def fit(self, X, y, optimization="GradientDescent"):
        if self.normalize:
            Xmax, Xmin = X.max(), X.min()
            X = (X - Xmin)/(Xmax - Xmin)
        if self.intercept:
            X = np.append(X, np.ones((X.shape[0],1)), axis=1)
            print(X.shape)

        self.W = np.zeros((X.shape[1],1))

        if optimization=="GradientDescent":
            optimizer = GradientDescent(X, y, self.model_name, W = self.W)
            self.W, epoch_losses = optimizer.optimize()
            self.trained=True

        return epoch_losses

    def predict(self, X):
        if not self.trained:
            raise Exception("Please train your model before generating predictions.")
        if self.intercept:
            X = np.append(X, np.ones((X.shape[0],1)), axis=1)
        y_pred = np.dot(X, self.W)
        return y_pred

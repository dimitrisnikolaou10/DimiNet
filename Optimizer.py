import numpy as np


class GradientDescent:
    def __init__(self, X, y, model_name, W = None, lr=0.001, epochs=100):
        self.X = X
        if W is not None:
            self.W = W
        self.y = y
        self.model_name = model_name
        self.lr = lr
        self.epochs = epochs
        self.samples = y.shape[0]

    def optimize(self, loss="mse"):
        epoch_losses = {}
        self.loss = loss
        for i in range(100000000):
            if self.model_name=="LinearRegression":
                error = self.LinearRegressionStep(i)
            elif self.model_name=="Lasso":
                error = self.LassoStep(i)
            elif self.model_name=="Ridge":
                error = self.RidgeStep(i)
            epoch_losses[i] = error
            if i % 100 == 0:
                print("Epoch {} error is {}.".format(i, error))
            # TODO printing of errors improvement
            self.lr *= 0.99999999 # learning rate decay (make this separate component)
            # TODO learning rate decay as component
            # TODO momentum as component
            # TODO early stop
        return self.W, epoch_losses

    def LinearRegressionStep(self, i):
        y_pred = self.__standard_product(self.W, self.X)
        losses = y_pred - self.y
        # print(losses)

        if self.loss == "mse":
            dw = 1/self.samples * np.dot(self.X.T, losses)
            dw = dw.reshape((self.W.shape[0],1))

            self.W -= self.lr*dw

            sum_of_squared_losses = np.sum(np.power(losses, 2))
            mse = sum_of_squared_losses/(self.samples*2) # doesn't take into account last

        return mse

    def LassoStep(self, i):
        # TODO create this properly
        y_pred = self.__standard_product(self.W, self.X)
        losses = y_pred - self.y

        if loss=="mse":
            dw = -2/self.samples * np.dot(self.X.T, (self.__standard_product(self.W, self.X) - self.y))

            self.W -= self.lr*dw

            sum_of_squared_losses = np.sum(np.power(losses, 2))
            mse = sum_of_squared_losses/self.samples

        return mse

    def RidgeStep(self, i):
        # TODO create this properly
        y_pred = self.__standard_product(self.W, self.X)
        losses = y_pred - self.y

        if loss=="mse":
            dw = -2/self.samples * np.sum(np.multiply(X,losses))

            self.W -= self.lr*dw

            sum_of_squared_losses = np.sum(np.power(losses, 2))
            mse = sum_of_squared_losses/self.samples

        return mse


    def __standard_product(self, W, X):
        return np.dot(self.X, self.W)

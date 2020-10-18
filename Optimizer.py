import numpy as np

# TODO Adam
class GradientDescent:
    def __init__(self, X, y, model_name, W = None, lr=0.001, epochs=np.power(10, 3)):
        self.X = X
        if W is not None:
            self.W = W
        self.y = y
        self.model_name = model_name
        self.lr = lr
        self.epochs = epochs
        self.samples = y.shape[0]
        self.gradients = []

    def optimize(self, loss="mse", print_loss_every=100, batch_size=None, momentum=0.1, lr_decay=0.000001, early_stop=0, lambda_1=None, lambda_2=None):
        # loss function to choose, every how many epochs to print, how big should batch size be,
        # how much should I weight old gradients, how much should the learning rate decay every epoch
        # Every how many epochs should we check for early stop criteria (Note 3 check fails and we stop)
        epoch_losses = {}
        early_stop_count = 0
        self.loss = loss
        if batch_size:
            batch_indices = [x for x in range(self.samples) if x % batch_size == 0]
        for i in range(self.epochs):
            if batch_size:
                batch_errors = []
                for index in batch_indices:
                    X = self.X[index:index + batch_size]
                    y = self.y[index:index + batch_size]
                    error = self.__ModelStep(X, y, momentum, lambda_1, lambda_2)
                    batch_errors.append(error)
                error = np.mean(batch_errors)
            else:
                error = self.__ModelStep(self.X, self.y, momentum)
            epoch_losses[i] = error
            if i % print_loss_every == 0:
                print("Epoch {} error is {}.".format(i, error))
            if early_stop != 0 and i % early_stop == 0 and i > 0:
                if error <= epoch_losses[-1]:
                    early_stop_count += 1
                else:
                    early_stop_count = 0
                if early_stop_count >= 3:
                    break
            self.lr *= (1-lr_decay)  # learning rate decay (make this separate component)
        return self.W, epoch_losses

    def __ModelStep(self, X, y, momentum, lambda_1=None, lambda_2=None):
        if self.model_name == "LinearRegression":
            error = self.LinearRegressionStep(X, y, momentum)
        elif self.model_name == "Lasso":
            error = self.LassoStep(X, y, momentum, lambda_1)
        elif self.model_name == "Ridge":
            error = self.RidgeStep(X, y, momentum, lambda_2)
        return error


    def LinearRegressionStep(self, X, y, momentum):
        y_pred = self.__standard_product(self.W, X)
        losses = y_pred - y

        if self.loss == "mse":
            dw = 1/self.samples * np.dot(X.T, losses)
            dw = dw.reshape((self.W.shape[0], 1))
            if self.gradients:
                mdw = momentum*self.gradients[-1] + (1-momentum)*dw
            else:
                mdw = dw

            self.gradients.append(mdw)

            self.W -= self.lr*mdw

            sum_of_squared_losses = np.sum(np.power(losses, 2))
            mse = sum_of_squared_losses/(self.samples*2)  # doesn't take into account last

        return mse

    def LassoStep(self, X, y, momentum, lambda_1):
        y_pred = self.__standard_product(self.W, X)
        losses = y_pred - y

        if self.loss == "mse":
            dw = 1 / self.samples * np.dot(X.T, losses) + lambda_1 * self.W
            dw = dw.reshape((self.W.shape[0], 1))
            if self.gradients:
                mdw = momentum * self.gradients[-1] + (1 - momentum) * dw
            else:
                mdw = dw

            self.gradients.append(mdw)

            self.W -= self.lr * mdw

            sum_of_squared_losses = np.sum(np.power(losses, 2))
            mse = sum_of_squared_losses / (self.samples * 2)  # doesn't take into account last

        return mse

    def RidgeStep(self, X, y, momentum, lambda_2):
        y_pred = self.__standard_product(self.W, X)
        losses = y_pred - y

        if self.loss == "mse":
            dw = 1/self.samples * np.dot(X.T, losses) + lambda_2 * self.W
            dw = dw.reshape((self.W.shape[0], 1))
            if self.gradients:
                mdw = momentum*self.gradients[-1] + (1-momentum)*dw
            else:
                mdw = dw

            self.gradients.append(mdw)

            self.W -= self.lr*mdw

            sum_of_squared_losses = np.sum(np.power(losses, 2))
            mse = sum_of_squared_losses/(self.samples*2)  # doesn't take into account last

        return mse

    def LogisticRegressionStep(self, X, y, momentum):
        h = self.__sigmoid_function(self.W, X)
        loss = np.dot(-y, np.log(h)) - np.dot((1-y), (1-h))

        dw = 1/self.samples * np.dot(X.T, h - y)
        dw = dw.reshape((self.W.shape[0], 1))
        if self.gradients:
            mdw = momentum*self.gradients[-1] + (1-momentum)*dw
        else:
            mdw = dw

        self.gradients.append(mdw)

        self.W -= self.lr*mdw

        return loss

    def __standard_product(self, W, X):
        return np.dot(X, W)

    def __sigmoid_function(self, W, X):
        return 1 / (1 + np.exp(-np.dot(W, X)))

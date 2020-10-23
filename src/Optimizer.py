import numpy as np
import utils

# TODO Adam
# TODO general cleanup
# TODO Unit tests
# TODO Work in an environment, set up dependencies

class GradientDescent:
    """
    Common Optimisation method.
    Details : https://towardsdatascience.com/understanding-the-mathematics-behind-gradient-descent-dde5dc9be06e
    Allows for adaptive learning rate, mini-batch training, early stop, momentum and regularisation.
     - X: The training data (numpy matrix)
     - y: The labels for the training data (numpy array)
     - model_name: The name of the model being optimised (string)
     - W: The initial weights where the descent starts from (numpy matrix) [OPTIONAL]
     - lr: Th learing rate (float) [OPTIONAL]
     - epochs: Number of times we will run over our training set (integer) [OPTIONAL]
    """
    def __init__(self, X, y, model_name, W=None, lr=0.001, epochs=np.power(10, 7)):
        self.X = X
        if W is not None:
            self.W = W
        self.y = y
        self.model_name = model_name
        self.lr = lr
        self.epochs = epochs
        self.samples = y.shape[0]
        self.gradients = []

    def optimize(self, loss="mse", print_loss_every=100, batch_size=None, momentum=0, lr_decay=0.000001, early_stop=0, lambda_2=None):
        """
        Method of Gradient Descent class that runs the optimization.
        - loss: the loss function to be used (string) [OPTIONAL]
        - print_loss_every: every how many epochs we should print the loss (integer) [OPTIONAL]
        - batch_size: size of the batch that a training step will require (integer) [OPTIONAL]
        - momentum: Extent at which past gradient will influece the current one (float) [OPTIONAL]
        - lr_decay: % drop of learning rate on each run (float) [OPTIONAL]
        - early_stop: every how many epochs should we check if it's time to stop (integer) [OPTIONAL]
        - lambda_2: Ridge regularization parameter (float) [OPTIONAL]
        """
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
                    error = self.__ModelStep(X, y, momentum, lambda_2)
                    batch_errors.append(error)
                error = np.mean(batch_errors)
            else:
                error = self.__ModelStep(self.X, self.y, momentum)
            epoch_losses[i] = error
            if i % print_loss_every == 0:
                print("Epoch {} error is {}.".format(i, error))
            if early_stop != 0 and i % early_stop == 0 and i > 0:  # check every early_stop epochs if we should stop
                if error <= epoch_losses[-1]:
                    early_stop_count += 1
                else:
                    early_stop_count = 0
                if early_stop_count >= 3:
                    break
            self.lr *= (1-lr_decay)
        return self.W, epoch_losses

    def __ModelStep(self, X, y, momentum, lambda_2=None):
        """
        Function that is not a method. Same for the specific steps below.
        """
        if self.model_name == "LinearRegression":
            error = self.LinearRegressionStep(X, y, momentum)
        elif self.model_name == "RidgeRegression":
            error = self.RidgeStep(X, y, momentum, lambda_2)
        elif self.model_name == "LogisticRegression":
            error = self.LogisticRegressionStep(X, y, momentum)
        elif self.model_name == "RidgeClassification":
            error = self.RidgeClassificationStep(X, y, momentum, lambda_2)
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


    def RidgeRegressionStep(self, X, y, momentum, lambda_2):
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

            sum_of_squared_losses = np.sum(np.power(losses, 2)) + lambda_2*np.dot(self.W.T, self.W)
            mse = sum_of_squared_losses/(self.samples*2)  # doesn't take into account last

        return mse

    def LogisticRegressionStep(self, X, y, momentum):
        h = utils.sigmoid_function(self.W, X)
        loss = -1/self.samples * (np.dot((y.T), np.log(h)) + np.dot((1-y).T, (1-h)))

        dw = 1/self.samples * np.dot(X.T, h - y)
        dw = dw.reshape((self.W.shape[0], 1))
        if self.gradients:
            mdw = momentum*self.gradients[-1] + (1-momentum)*dw
        else:
            mdw = dw

        self.gradients.append(mdw)

        self.W -= self.lr*mdw

        return loss


    def RidgeClassificationStep(self, X, y, momentum, lambda_2):
        h = utils.sigmoid_function(self.W, X)
        loss = -1 / self.samples * (np.dot((y.T), np.log(h)) + np.dot((1 - y).T, (1 - h))) + lambda_2*np.dot(self.W.T, self.W)

        dw = 1 / self.samples * np.dot(X.T, h - y) + lambda_2 * self.W
        dw = dw.reshape((self.W.shape[0], 1))
        if self.gradients:
            mdw = momentum * self.gradients[-1] + (1 - momentum) * dw
        else:
            mdw = dw

        self.gradients.append(mdw)

        self.W -= self.lr * mdw

        return loss

    def __standard_product(self, W, X):
        return np.dot(X, W)


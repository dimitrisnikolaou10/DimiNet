import matplotlib.pyplot as plt
import numpy as np


def plot_errors(errors): # expecting errors to be in dict
    x_axis = list(errors.keys())
    values = list(errors.values())
    plt.plot(x_axis, values, '-')
    # plt.ylim((2500,10000))
    plt.show()


def mean_squared_error(errors):  # expecting errors to be in dict
    mse = np.mean(np.square(errors))
    print("MSE is equal to {}".format(mse))
    return mse


def mean_absolute_error(errors):  # expecting errors to be in dict
    mae = np.mean(np.abs(errors))
    print("MAE is equal to {}".format(mae))
    return mae

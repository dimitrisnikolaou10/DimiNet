import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def accuracy(y, y_pred):  # numpy arrays as arguments
    match = 0
    all_examples = y.shape[0]
    for y, p in zip(y, y_pred):
        if int(y) == int(p):
            match += 1
    acc = match/all_examples
    print("Accuracy is equal to {}".format(acc))
    return acc


def confusion_matrix(y, y_pred):  # numpy arrays as arguments
    y_actu = pd.Series(np.squeeze(y), name='Actual')
    y_pred = pd.Series(np.squeeze(y_pred), name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)
    return df_confusion

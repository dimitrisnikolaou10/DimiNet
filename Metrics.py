import matplotlib.pyplot as plt

def plot_errors(errors): # expecting errors to be in dict
    x_axis = list(errors.keys())
    values = list(errors.values())
    plt.plot(x_axis, values, '-')
    plt.ylim((2500,10000))
    plt.show()

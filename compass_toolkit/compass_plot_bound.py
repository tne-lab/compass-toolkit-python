import matplotlib.pyplot as plt
import numpy as np


def compass_plot_bound(mode, x, y, y_low, y_high, x_title='', y_title='', fig_title='', color=[0.9, 0.9, 0.9]):
    if len(y_high) != len(y_low) or len(y_low) != len(x) or len(x) != len(y):
        print('Inconsistent input sizes, quitting')
        return

    # plots the data plus its bound
    x_combined = np.concatenate((x, x[::-1]))
    y_combined = np.concatenate((y_high, y_low[::-1]))
    plt.fill(x_combined, y_combined, color=color, edgecolor=color)
    h = plt.plot(x, y, 'k', linewidth=3)

    return h

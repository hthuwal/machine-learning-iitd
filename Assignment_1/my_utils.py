import numpy as np
import pandas as pd


def normalize(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return np.apply_along_axis(lambda x: (x - mean) / std, 1, x)


def category_to_discretevalues(Y):
    classes = {}
    y = []
    label = 0
    for each in Y:
        if each not in classes:
            classes[each] = label
            label += 1

        y.append(classes[each])
    return np.array(y), classes


def read_files(x_file, y_file, sep=','):
    X = pd.read_csv("dataset/" + x_file, header=None, sep=sep)
    X = X.as_matrix()
    X = normalize(X)
    temp = X.flatten()
    std = np.std(temp)
    xlim = (temp[np.argmin(temp)] - std, temp[np.argmax(temp)] + std)
    X = np.insert(X, 0, 1.0, axis=1)

    Y = pd.read_csv("dataset/" + y_file, header=None, sep=sep)
    Y = Y.as_matrix().flatten()

    if (isinstance(Y[0], str)):
        Y, cls_labels = category_to_discretevalues(Y)
        std = np.std(Y)
        ylim = (Y[np.argmin(Y)] - std, Y[np.argmax(Y)] + std)
        return X, Y, xlim, ylim, cls_labels

    std = np.std(Y)

    ylim = (Y[np.argmin(Y)] - std, Y[np.argmax(Y)] + std)

    return X, Y, xlim, ylim

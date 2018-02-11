import numpy as np
import my_utils


def g(z):
    return 1 / (1 + np.exp(-z))

# def gradient_ltheta(x, y, theta):


data = my_utils.read_files("logisticX.csv", "logisticY.csv")
X = data[0]
Y = data[1]
std = np.std(X, axis=0)
argmax = np.argmax(X, axis=0)
argmin = np.argmin(X, axis=0)

x1_lim = (X[argmin[1]][1] - std[1], X[argmax[1]][1] + std[1])
x2_lim = (X[argmin[2]][2] - std[2], X[argmax[2]][2] + std[2])

import my_utils
import numpy as np


def get_phi():
    return num_yi_is_1 / (num_yi_is_1 + num_yi_is_0)


def get_mu0():
    num_features = X.shape[1] - 1
    mu0 = np.zeros([num_features + 1, ])

    for x, y in zip(X, Y):
        mu0 = mu0 + x * (1 if y == 0 else 0)
    mu0 = mu0 / num_yi_is_0

    return mu0


def get_mu1():
    num_features = X.shape[1] - 1
    mu1 = np.zeros([num_features + 1, ])

    for x, y in zip(X, Y):
        mu1 = mu1 + x * (1 if y == 1 else 0)
    mu1 = mu1 / num_yi_is_1

    return mu1


data = my_utils.read_files("q4x.dat", "q4y.dat", sep='\s+')
X = data[0]
X = np.delete(X, 0, axis=1)  # in gda there is no intercept term
Y = data[1]
cls_labels = data[4]
std = np.std(X, axis=0)
argmax = np.argmax(X, axis=0)
argmin = np.argmin(X, axis=0)

x1_lim = (X[argmin[0]][0] - std[0], X[argmax[0]][0] + std[0])
x2_lim = (X[argmin[1]][1] - std[1], X[argmax[1]][1] + std[1])
num_yi_is_1 = np.sum(Y)  # because rest are zero so sum
num_yi_is_0 = len(Y) - num_yi_is_1

get_mu0()

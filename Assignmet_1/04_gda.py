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


def get_covariance(mu0, mu1, same=True):
    num_features = X.shape[1] - 1
    mu0.shape = [num_features + 1, 1]
    mu1.shape = [num_features + 1, 1]

    if same:
        sigma = np.zeros([num_features + 1, num_features + 1])
        for x, y in zip(X, Y):
            mu = mu0 if y == 0 else mu1
            x.shape = [num_features + 1, 1]
            sigma = sigma + (x - mu) @ (x - mu).T

        sigma = sigma / (num_yi_is_0 + num_yi_is_1)
        return sigma

    else:
        sigma0 = np.zeros([num_features + 1, num_features + 1])
        sigma1 = np.zeros([num_features + 1, num_features + 1])

        for x, y in zip(X, Y):
            x.shape = [num_features + 1, 1]
            sigma0 = sigma0 + ((x - mu0) @ (x - mu0).T) * (1 if y == 0 else 0)
            sigma1 = sigma1 + ((x - mu1) @ (x - mu1).T) * (1 if y == 1 else 0)

        sigma0 = sigma0 / num_yi_is_0
        sigma1 = sigma1 / num_yi_is_1
        return sigma0, sigma1


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

mu0, mu1 = get_mu0(), get_mu1()
sigma0, sigma1 = get_covariance(mu0, mu1, same=False)
print(sigma0, "\n", sigma1)

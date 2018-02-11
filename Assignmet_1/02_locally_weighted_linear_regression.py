import numpy as np
import pandas as pd


def normalize(x):
    mean = np.mean(x, axis=0)
    std = np.std(x)
    return np.apply_along_axis(lambda x: (x - mean) / std, 1, x)


def ulr_normal(x, y):
    theta = np.matrix(x.T @ x)
    theta = theta.I @ x.T @ y
    theta = np.squeeze(np.asarray(theta))
    return theta


X = pd.read_csv("dataset/linearX.csv", header=None)
X = X.as_matrix()
X = normalize(X)
temp = X.flatten()
std = np.std(temp)
xlim = (temp[np.argmin(temp)] - std, temp[np.argmax(temp)] + std)
X = np.insert(X, 0, 1.0, axis=1)


Y = pd.read_csv("dataset/linearY.csv", header=None)
Y = Y.as_matrix().flatten()
std = np.std(Y)
ylim = (Y[np.argmin(Y)] - std, Y[np.argmax(Y)] + std)

print(ulr_normal(X, Y))

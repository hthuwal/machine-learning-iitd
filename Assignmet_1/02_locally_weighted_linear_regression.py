import numpy as np
import my_utils


def normalize(x):
    mean = np.mean(x, axis=0)
    std = np.std(x)
    return np.apply_along_axis(lambda x: (x - mean) / std, 1, x)


def ulr_normal(x, y):
    theta = np.matrix(x.T @ x)
    theta = theta.I @ x.T @ y
    theta = np.squeeze(np.asarray(theta))
    return theta


X, Y, xlim, ylim = my_utils.read_files("weightedX.csv", "weightedY.csv")
theta_ulr_normal = ulr_normal(X, Y)

import numpy as np
import my_utils


def g(z):
    return 1 / (1 + np.exp(-z))


def gradient_ltheta(x, y, theta):
    num_examples = x.shape[0]
    num_features = x.shape[1] - 1
    gradient = np.zeros([num_features + 1, ])

    for jth_feature in range(0, num_features + 1):

        gradient_wrt_jth_feature = 0.0

        for ith_example in range(0, num_examples):
            gradient_wrt_jth_feature += (y[ith_example] - g(theta @ x[ith_example])).flatten()[0] * x[ith_example][jth_feature]

        gradient[jth_feature] = gradient_wrt_jth_feature

    return gradient


def hessian_ltheta(x, y, theta):
    num_examples = x.shape[0]
    num_features = x.shape[1] - 1

    hessian = np.zeros([num_features + 1, num_features + 1])

    for ith_feature in range(0, num_features + 1):
        for jth_feature in range(0, num_features + 1):

            hessian[ith_feature][jth_feature] = 0.0

            for kth_example in range(0, num_examples):

                gz = g(theta @ x[kth_example])
                hessian[ith_feature][jth_feature] += x[kth_example][ith_feature] * x[kth_example][jth_feature] * (gz - 1) * (gz)

    return hessian

data = my_utils.read_files("logisticX.csv", "logisticY.csv")
X = data[0]
Y = data[1]
std = np.std(X, axis=0)
argmax = np.argmax(X, axis=0)
argmin = np.argmin(X, axis=0)

x1_lim = (X[argmin[1]][1] - std[1], X[argmax[1]][1] + std[1])
x2_lim = (X[argmin[2]][2] - std[2], X[argmax[2]][2] + std[2])

print(gradient_ltheta(X, Y, np.array([1, 2, 3])))
print(hessian_ltheta(X, Y, np.array([1, 2, 3])))

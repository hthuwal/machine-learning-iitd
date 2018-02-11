import numpy as np
import matplotlib.pyplot as plt
import my_utils


def hypothesis_plot_ulr(x, y, theta, legend):
    ax = plt.subplot(1, 2, 1)
    ax.set_title('Hypothesis Function and Scatter Plot')

    x_line = np.linspace(xlim[0], xlim[1], 200)
    x_line.shape = [200, 1]
    x_line = np.insert(x_line, 0, 1.0, axis=1)
    y_line = np.matmul(theta, np.transpose(x_line))

    x = np.delete(x, 0, axis=1)
    x_line = np.delete(x_line, 0, axis=1)

    plt.xlabel("Acidity")
    plt.ylabel("Density")
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.scatter(x, y)
    hypothesis_function, = plt.plot(x_line, y_line, '#FF4500')
    ax.legend([legend])


def normalize(x):
    mean = np.mean(x, axis=0)
    std = np.std(x)
    return np.apply_along_axis(lambda x: (x - mean) / std, 1, x)


def ulr_normal(x, y):
    theta = np.matrix(x.T @ x)
    theta = theta.I @ x.T @ y
    theta = np.squeeze(np.asarray(theta))
    return theta


def get_weight_matrix(x, tau):
    xis = np.delete(X, 0, axis=1)
    xis = (-1 * ((x - xis)**2)) / (2 * tau * tau)
    xis = np.exp(xis)
    xis.shape = [100, ]
    xis = np.diag(xis)
    return xis


X, Y, xlim, ylim = my_utils.read_files("weightedX.csv", "weightedY.csv")
theta_ulr_normal = ulr_normal(X, Y)

get_weight_matrix(4, 5)
hypothesis_plot_ulr(X, Y, theta_ulr_normal, "theta: %s" % (theta_ulr_normal))

plt.show()

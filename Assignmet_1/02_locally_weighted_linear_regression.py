import numpy as np
import matplotlib.pyplot as plt
import my_utils


def hypothesis_plot_ulr(x, y, theta):
    ax = plt.subplot(1, 2, 1)
    ax.set_title('Analytical Solution: Unweighted Linear Regression')

    x_line = np.linspace(xlim[0], xlim[1], 200)
    x_line.shape = [200, 1]
    x_line = np.insert(x_line, 0, 1.0, axis=1)
    y_line = np.matmul(theta, np.transpose(x_line))

    x = np.delete(x, 0, axis=1)
    x_line = np.delete(x_line, 0, axis=1)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.scatter(x, y)
    hypothesis_function, = plt.plot(x_line, y_line, '#FF4500')
    ax.legend(["theta: %s" % (theta_ulr_normal)])


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


def wlr_normal(x, y, cur_x, tau):
    W = get_weight_matrix(cur_x, tau)
    theta = np.matrix(x.T @ W @ x)
    theta = theta.I @ x.T @ W @ y
    theta = np.squeeze(np.asarray(theta))
    return theta


def get_y(x, theta):
    x = np.array([1, x])
    return theta.T @ x


def hypothesis_plot_wlr(x, y, tau):
    ax = plt.subplot(1, 2, 2)
    ax.set_title('Analytical Solution: Weighted Linear Regression')

    x_wlr = np.linspace(xlim[0], xlim[1], 200)
    x_wlr.shape = [200, 1]
    y_wlr = np.array([get_y(cur_x, wlr_normal(x, y, cur_x, tau)) for cur_x in x_wlr])

    x = np.delete(x, 0, axis=1)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.scatter(x, y)  # original points
    plt.scatter(x_wlr, y_wlr, s=3)  # curve resultant of
    plt.legend(["tau: %0.4f" % tau])


def fun(x, y):
    taus = 1 / np.linspace(0, 10, 50)
    for tau in taus:
        # print(tau)
        fig.clf()
        hypothesis_plot_ulr(X, Y, theta_ulr_normal)
        hypothesis_plot_wlr(x, y, tau)
        plt.pause(0.2)


X, Y, xlim, ylim = my_utils.read_files("weightedX.csv", "weightedY.csv")
theta_ulr_normal = ulr_normal(X, Y)

# fig = plt.figure(figsize=(1920 / 96, 1080 / 96), dpi=96)  # forcing it to be of size 1920x1080
fig = plt.figure()

mng = plt.get_current_fig_manager()
# mng.full_screen_toggle()
mng.resize(*mng.window.maxsize())

hypothesis_plot_ulr(X, Y, theta_ulr_normal)
tau = 10
# fun(X, Y)
hypothesis_plot_wlr(X, Y, tau)

plt.pause(0.2)
plt.savefig("wlr/" + "tau_%0.2f.png" % tau)
plt.show()

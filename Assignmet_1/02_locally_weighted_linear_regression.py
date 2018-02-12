import numpy as np
import matplotlib.pyplot as plt
import my_utils


def hypothesis_plot_ulr(x, y, theta):
    """
    Hypothesis plot of the analytical solution of the Unweighted Linear Regression
    """
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


def ulr_normal(x, y):
    """

    Analytical Solution of unweighted linear regression

    Arguments:
        x {np.ndarray} -- [Design Matrix]
        y {np.ndarray} -- [Outputs]

    Returns:
        [theta] -- [analytical solution]
        (xtranspos * x)^(-1) * xtanspos * y
    """
    theta = np.matrix(x.T @ x)
    theta = theta.I @ x.T @ y
    theta = np.squeeze(np.asarray(theta))
    return theta


def get_weight_matrix(x, tau):
    """
    Calculate weight matrix based on value of tau and x(local point)
    """

    xis = np.delete(X, 0, axis=1)  # remove intercept term

    # calculating weight matrix
    xis = (-1 * ((x - xis)**2)) / (2 * tau * tau)
    xis = np.exp(xis)
    xis.shape = [100, ]
    xis = np.diag(xis)  # converting into diagonal matrix
    return xis


def wlr_normal(x, y, cur_x, tau):
    """
    Analytical Solution of weighted Linear Regression
    (X.tanspos W X)^(-1) X.transpos W Y
    """

    W = get_weight_matrix(cur_x, tau)
    theta = np.matrix(x.T @ W @ x)
    theta = theta.I @ x.T @ W @ y
    theta = np.squeeze(np.asarray(theta))
    return theta


def get_y(x, theta):
    """
    Function to get y corresponding to x based on theta
    """
    x = np.array([1, x])
    return theta.T @ x


def hypothesis_plot_wlr(x, y, tau):
    """
    Function to plot the hypthess function of weighted linear regression
    """
    ax = plt.subplot(1, 2, 2)
    ax.set_title('Analytical Solution: Weighted Linear Regression')

    x_wlr = np.linspace(xlim[0], xlim[1], 200)
    x_wlr.shape = [200, 1]

    # Find weighted linear regression solution around each x and get corresponding y according to
    # the wlr solution
    y_wlr = np.array([get_y(cur_x, wlr_normal(x, y, cur_x, tau)) for cur_x in x_wlr])

    x = np.delete(x, 0, axis=1)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.scatter(x, y)  # original points
    plt.scatter(x_wlr, y_wlr, s=3)
    plt.legend(["tau: %0.4f" % tau])


def fun(x, y):
    """
    Dynamically update Tau and show corresponding plots
    """
    taus = 1 / np.linspace(0, 10, 50)
    for tau in taus:
        # print(tau)
        fig.clf()
        hypothesis_plot_ulr(X, Y, theta_ulr_normal)
        hypothesis_plot_wlr(x, y, tau)
        plt.pause(0.2)


#read files
X, Y, xlim, ylim = my_utils.read_files("weightedX.csv", "weightedY.csv")
theta_ulr_normal = ulr_normal(X, Y)

# fig = plt.figure(figsize=(1920 / 96, 1080 / 96), dpi=96)  # forcing it to be of size 1920x1080
fig = plt.figure()

mng = plt.get_current_fig_manager()
# mng.full_screen_toggle()
mng.resize(*mng.window.maxsize())

hypothesis_plot_ulr(X, Y, theta_ulr_normal)
tau = 0.8
# fun(X, Y)
hypothesis_plot_wlr(X, Y, tau)

plt.pause(0.2)
# plt.savefig("Plots/wlr/" + "tau_%0.2f.png" % tau)
plt.show()

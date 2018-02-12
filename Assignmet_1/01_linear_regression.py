from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import my_utils

legend = """
    iteration: %d
    eeta: %g
    theta: %0.16f %0.16f
    loss_function: %s
    threshold: %g
    loss: %0.16f
    """


def hypothesis_plot(x, y, subplot=True):
    """

    Scatter Plots the points. and and the initial hypothesis funtion
    with theta = vector of zeros

    Arguments:
        x {[np.ndarray]} -- [Design Matrix X]
        y {[np.ndarray]} -- [outputs]

    Keyword Arguments:
        subplot {bool} -- [if True all three dynamic plots are done on a single figure
        as subplots. Otherwise three seperate figures are created] (default: {True})

    Returns:
        [fig axis, hypothesis function object]
    """
    if subplot:
        global fig
        ax = fig.add_subplot(gs[0, :])
    else:
        fig = plt.figure(1)
        ax = plt.subplot(1, 1, 1)

    x_line = np.linspace(xlim[0], xlim[1], 200)
    x_line.shape = [200, 1]
    x_line = np.insert(x_line, 0, 1.0, axis=1)
    theta = np.zeros([x.shape[1], ])
    y_line = np.matmul(theta, np.transpose(x_line))

    x = np.delete(x, 0, axis=1)
    x_line = np.delete(x_line, 0, axis=1)

    plt.xlabel("Acidity")
    plt.ylabel("Density")
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.scatter(x, y)  # scatter plot of the dataset
    hypothesis_function, = plt.plot(x_line, y_line, '#FF4500')  # line plot of initial hypthesis function

    if not subplot:
        return fig, hypothesis_function, ax

    return hypothesis_function, ax


def update_hypothesis_plot(theta, cur_legend):
    """

    Updates the hypothesis function in the hypothesis plot based on current theta

    Arguments:
        theta {[np.ndarray]} -- [theta vector]
        cur_legend {[str]} -- [legend to be shown on graph]
    """
    x_line = np.linspace(xlim[0], xlim[1], 200)
    x_line.shape = [200, 1]
    x_line = np.insert(x_line, 0, 1.0, axis=1)
    y_line = np.matmul(theta, np.transpose(x_line))  # calculating y correspoding to x based on new theta

    hthetax.set_ydata(y_line)  # updating the hypothesis function
    hplot.legend([cur_legend])  # updating the legend function


def plot_error_surface(subplot=True):
    """

    Plots the 3d error surface, with the initial value of error.

    Keyword Arguments:
          subplot {bool} -- [if True all three dynamic plots are done on a single figure
          as subplots. Otherwise three seperate figures are created] (default: {True})

    Returns:
        [fig axis and surface object]
    """
    if subplot:
        global fig
        ax = fig.add_subplot(gs[1, 0], projection='3d')
    else:
        fig = plt.figure(2)
        ax = plt.subplot(1, 1, 1, projection='3d')

    ax.set_title("3D surface of Error Function")

    # theta 0, theta 1, jtheta are global variables thatcorrespond to mesh grid and the error
    # function evaluated at the end
    surf = ax.plot_surface(theta0, theta1, jtheta, cmap='viridis')  # 3d surface plot
    ax.set_zlim(-1, 100)
    ax.set_xlabel('theta0')
    ax.set_ylabel('theta1')
    ax.set_zlabel('jtheta')

    ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.01f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.01f'))

    init_theta = np.zeros([X.shape[1], ])
    init_error = mean_squared_error(init_theta)  # initial error

    # plotting the current error
    error_3d, = ax.plot([0], [0], [init_error], marker='o', markersize=3, color="#FF4500")

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    if not subplot:
        return fig, error_3d, ax

    return error_3d, ax


def update_error_surface(theta):
    """

    Updates the cur_error on 3d surface based on current theta

    Arguments:
        theta {[np.ndarray]} -- [theta vector]
    """
    xs, ys, zs = error_3d._verts3d
    xs = np.append(xs, theta[0])
    ys = np.append(ys, theta[1])
    zs = np.append(zs, mean_squared_error(theta))  # updating current_error

    # updating error_plot
    error_3d.set_xdata(xs)
    error_3d.set_ydata(ys)
    error_3d.set_3d_properties(zs)


def plot_error_contours(subplot=True):
    """

    Plots the initial contour 3d error surface, with the initial value of error.

    Keyword Arguments:
          subplot {bool} -- [if True all three dynamic plots are done on a single figure
          as subplots. Otherwise three seperate figures are created] (default: {True})

    Returns:
        [fig axis and contour object]
    """
    if subplot:
        global fig
        ax = fig.add_subplot(gs[1, 1])
    else:
        fig = plt.figure(3)
        ax = plt.subplot(1, 1, 1)

    ax.set_title("Contours of Error Function")
    cs = ax.contour(theta0, theta1, jtheta, 1)  # plot the contour  with just 1 level (for initialization)

    ax.set_xlabel('theta0')
    ax.set_ylabel('theta1')

    ax.xaxis.set_major_formatter(FormatStrFormatter('%.01f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.01f'))

    init_theta = np.zeros([X.shape[1], ])
    init_error = mean_squared_error(init_theta)

    # plotting the current error
    error_cs, = ax.plot([0], [0], marker='o', color="#FF4500", linestyle="None")

    if not subplot:
        return fig, error_cs, cs, ax

    return error_cs, cs, ax


def update_error_contour(theta):
    """

    Updates the cur_error on contour plot based on current theta

    Arguments:
        theta {[np.ndarray]} -- [theta vector]

    """
    levels = np.append(cs.levels, mean_squared_error(theta))
    cs_plot.contour(theta0, theta1, jtheta, np.sort(levels))

    # updating error plot
    error_cs.set_xdata(np.append(error_cs.get_xdata(), theta[0]))
    error_cs.set_ydata(np.append(error_cs.get_ydata(), theta[1]))


def mean_squared_error(theta):
    """
    Function to calculate the mean squared error that is J(theta)

    Arguments:
        theta {[np.ndarray]} -- [theta vector]

    Returns:
        [np.float64] -- [J(theta)]
    """
    j_theta = 0.5 * np.sum(np.square(Y - theta @ np.transpose(X)))
    return j_theta


def change_in_theta(theta, old_theta):
    """
    Loss function: maximum absolute change in thetas
    """
    difference = np.absolute(np.subtract(theta, old_theta))
    return difference[np.argmax(difference)]


def change_in_error(theta, old_theta):
    """
    Loss function: Change in error (J(theta))
    """
    return abs(mean_squared_error(theta) - mean_squared_error(old_theta))


def update_plots(theta, cur_legend):
    """
    Funnction tat updates all plots
    """
    update_error_surface(theta)
    update_error_contour(theta)
    update_hypothesis_plot(theta, cur_legend)


def bgd(x, y, eeta, max_iter, threshold, loss_function="change_in_theta"):
    """Gradient Descent

    Implements gradient descent Algorithm

    Arguments:
        x  -- [Design Matrix]
        y  -- [outputs]
        eeta -- learning rate
        max_iter -- maximum numbe of iterations allowed
        threshold -- threshold for loss function

    Keyword Arguments:
        loss_function {str} -- [loss function to be used] (default: {"change_in_theta"})
    """
    num_examples = x.shape[0]
    num_features = x.shape[1] - 1

    theta = np.zeros([num_features + 1, ])  # initializing theta with zeros
    old_theta = np.zeros([num_features + 1, ])  # to save previous value of theta (for one of the loss functions)
    gradient = np.zeros([num_features + 1, ])  # to save the gradient evaluated (for one of the loss functions)

    update_plots(theta, "Initializing")

    iter = 0
    while True:
        iter += 1

        # Update theta j (each parameter)
        for jth_feature in range(0, num_features + 1):

            gradient_wrt_jth_feature = 0.0

            # summation over all examples
            for ith_example in range(0, num_examples):
                gradient_wrt_jth_feature += (y[ith_example] - theta @ x[ith_example]).flatten()[0] * x[ith_example][jth_feature]

            gradient[jth_feature] = gradient_wrt_jth_feature  # saving the gradient of jth feature for loss function
            theta[jth_feature] += eeta * gradient_wrt_jth_feature  # updating theta j

        # calculate loss based on the type of loss function
        if loss_function == "change_in_theta":
            loss = change_in_theta(theta, old_theta)
        elif loss_function == "change_in_error":
            loss = change_in_error(theta, old_theta)
        elif loss_function == "error":
            loss = mean_squared_error(theta)
        elif loss_function == "gradient":
            gradient = np.abs(gradient)
            loss = gradient[np.argmax(gradient)]

        print(iter, theta, loss)

        cur_legend = legend % (iter, eeta, theta[0], theta[1], loss_function, threshold, loss)  # update legend
        update_plots(theta, cur_legend)  # update plot
        plt.pause(0.02)
        # plt.savefig("bgd/%d.png" %iter)

        if (loss < threshold or iter == max_iter):
            break

        old_theta = np.array(theta)

    print("GDA solution")
    print(legend % (iter, eeta, theta[0], theta[1], loss_function, threshold, loss))


# read file
X, Y, xlim, ylim = my_utils.read_files("linearX.csv", "linearY.csv")

# initialize 3d grid for surface and contour plots
theta0 = np.linspace(0, 2, 100)
theta1 = np.linspace(-1, 1, 100)
theta0, theta1 = np.meshgrid(theta0, theta1)
jtheta = np.zeros(theta0.shape)

# TODO do this using some numpy trick
for i in range(0, len(jtheta)):
    for j in range(0, len(jtheta[0])):
        jtheta[i][j] = mean_squared_error(
            np.array([theta0[i][j], theta1[i][j]]))

# grid for subplots
gs = gridspec.GridSpec(2, 2)


subplot = True

if subplot:
    fig = plt.figure(1)
    hthetax, hplot = hypothesis_plot(X, Y, subplot=subplot)
    error_3d, surf_plot = plot_error_surface(subplot=subplot)
    error_cs, cs, cs_plot = plot_error_contours(subplot=subplot)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

else:
    fig1, hthetax, hplot = hypothesis_plot(X, Y, subplot=subplot)
    fig2, error_3d, surf_plot = plot_error_surface(subplot=subplot)
    fig3, error_cs, cs, cs_plot = plot_error_contours(subplot=subplot)


# bgd(X, Y, 0.001, 100, 0.0000000001, loss_function="change_in_theta")
# bgd(X, Y, 0.001, 100, 0.0000001, loss_function="gradient")
# bgd(X, Y, 0.01, 50000, 0.000119480, loss_function="error")

eeta = 0.019
bgd(X, Y, eeta, 100, 1.1e-5, loss_function="change_in_error") # calling gradient descent

# if subplot:
#     fig.savefig("Plots/bgd/bgd_b_c_d_%g.png" % eeta)
# else:
#     fig1.savefig("Plots/bgd/bgd_b.png")
#     fig2.savefig("Plots/bgd/bgd_c.png")
#     fig3.savefig("Plots/bgd/bgd_d.png")

plt.show()

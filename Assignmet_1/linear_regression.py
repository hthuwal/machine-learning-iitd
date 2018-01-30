import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

legend = """
    iteration: %d
    eeta: %g
    theta: %s
    loss_function: %s
    threshold: %g
    loss: %g
    """


def plot(x, y, theta):
    
    x_line = np.linspace(0,xlim[1],200)
    x_line.shape = [200,1]
    x_line = np.insert(x_line, 0, 1.0, axis=1)
    
    y_line = np.matmul(theta, np.transpose(x_line))

    x = np.delete(x, 0, axis=1)
    x_line = np.delete(x_line, 0, axis=1)

    plt.xlabel("Acidity")
    plt.ylabel("Density")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.scatter(x, y)
    plt.plot(x_line, y_line,'#FF4500')    
    plt.draw()


def mean_squared_error(theta):
    j_theta = 0.5 * np.sum(np.square(Y - theta @ np.transpose(X)))
    return j_theta
    # or
    # z = np.linalg.norm(y - theta @ np.transpose(x))
    # j_theta = 0.5 * z * z


def change_in_theta(theta, old_theta):
    difference = np.absolute(np.subtract(theta, old_theta))
    return difference[np.argmax(difference)]


def change_in_error(theta, old_theta):
    return abs(mean_squared_error(theta) - mean_squared_error(old_theta))


def bgd(x, y, eeta, max_iter, threshold, loss_function="change_in_theta"):
    num_examples = x.shape[0]
    num_features = x.shape[1] - 1

    theta = np.zeros([num_features+1, ])
    old_theta = np.zeros([num_features+1, ])
    gradient = np.zeros([num_features+1, ])

    iter = 0
    while True:
        iter += 1

        for jth_feature in range(0, num_features + 1):

            gradient_wrt_jth_feature = 0.0

            for ith_example in range(0, num_examples):
                gradient_wrt_jth_feature += (y[ith_example] - theta @ x[ith_example]).flatten()[0] * x[ith_example][jth_feature]

            gradient[jth_feature] = gradient_wrt_jth_feature
            theta[jth_feature] += eeta * gradient_wrt_jth_feature

        if(iter % 100 == 0):
            plt.gcf().clear()
            plot(x, y, theta)
            plt.legend(
                [legend % (iter, eeta, str(theta), loss_function, threshold, loss)])
            plt.pause(0.01)

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

        if (loss < threshold or iter == max_iter):
            break

        old_theta = np.array(theta)

    plt.gcf().clear()
    plot(x, y, theta)

    plt.legend(
        [legend % (iter, eeta, str(theta), loss_function, threshold, loss)])
    plt.show()

X = pd.read_csv("dataset/linearX.csv", header=None)
X = X.as_matrix()
temp = X.flatten()
std = np.std(temp)
xlim = (temp[np.argmin(temp)]-std, temp[np.argmax(temp)]+std)
X = np.insert(X, 0, 1.0, axis=1)  # x0 =


Y = pd.read_csv("dataset/linearY.csv", header=None)
Y = Y.as_matrix().flatten()
std = np.std(Y)
ylim = (Y[np.argmin(Y)]-std, Y[np.argmax(Y)]+std)

# bgd(X, Y, 0.0001, 50000, 0.0000000001, loss_function="change_in_theta")
# bgd(X, Y, 0.0001, 50000, 0.00001, loss_function="gradient")
bgd(X, Y, 0.0001, 50000, 0.000119480, loss_function="error")
# bgd(X, Y, 0.0001, 50000, 0.000000000001, loss_function="change_in_error")

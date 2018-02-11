import numpy as np
import matplotlib.pyplot as plt
import my_utils

legend = """
    iteration: %d
    theta: %s
    threshold: %g
    gradient(loss): %g
    """


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

    return np.matrix(hessian)


def newtons_method(x, y, max_iter, threshold):
    num_examples = x.shape[0]
    num_features = x.shape[1] - 1

    theta = np.zeros([num_features + 1, ])

    iter = 0

    while True:
        gradient = gradient_ltheta(x, y, theta)

        temp = np.abs(gradient)
        loss = temp[np.argmax(gradient)]

        print(iter, theta, loss)
        if(theta[2] != 0):
            update_decision_boundary_plot(theta, legend % (iter, theta, threshold, loss))

        if (loss < threshold or iter == max_iter):
            break

        theta = theta - np.array(hessian_ltheta(x, y, theta).I @ gradient)  # update
        theta.shape = [num_features + 1, ]
        iter += 1


def decision_boundary_plot(x, y):
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Scatter Plot and decision boundary')

    xone = x[:, 1]
    xtwo = x[:, 2]

    xone_yis1 = []
    xtwo_yis1 = []

    xone_yis0 = []
    xtwo_yis0 = []

    for x1, x2, y in zip(xone, xtwo, y):
        if y == 1:
            xone_yis1.append(x1)
            xtwo_yis1.append(x2)
        else:
            xone_yis0.append(x1)
            xtwo_yis0.append(x2)

    x1_line = np.linspace(x1_lim[0], x1_lim[1], 200)
    x1_line.shape = [200, 1]
    theta = np.zeros([x.shape[1], ])
    theta[2] = 1
    x2_line = np.array([-((theta[0] + theta[1] * x1) / theta[2]) for x1 in x1_line])

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim(x1_lim)
    plt.ylim(x2_lim)

    y0 = plt.scatter(xone_yis0, xtwo_yis0, marker='o')
    y1 = plt.scatter(xone_yis1, xtwo_yis1, marker='x')
    decision_boundary, = plt.plot(x1_line, x2_line, '#FF4500')

    return decision_boundary, ax, y0, y1


def update_decision_boundary_plot(theta, cur_legend):
    x1_line = np.linspace(x1_lim[0], x1_lim[1], 200)
    x1_line.shape = [200, 1]
    x2_line = np.array([-((theta[0] + theta[1] * x1) / theta[2]) for x1 in x1_line])
    db.set_ydata(x2_line)
    bplot.legend([y0, y1, db], ['class0', 'class1', cur_legend])
    plt.pause(2)


data = my_utils.read_files("logisticX.csv", "logisticY.csv")
X = data[0]
Y = data[1]
std = np.std(X, axis=0)
argmax = np.argmax(X, axis=0)
argmin = np.argmin(X, axis=0)

x1_lim = (X[argmin[1]][1] - std[1], X[argmax[1]][1] + std[1])
x2_lim = (X[argmin[2]][2] - std[2], X[argmax[2]][2] + std[2])


fig = plt.figure()

mng = plt.get_current_fig_manager()
# mng.full_screen_toggle()
mng.resize(*mng.window.maxsize())

db, bplot, y0, y1 = decision_boundary_plot(X, Y)
newtons_method(X, Y, 500, 1.0e-15)
plt.show()

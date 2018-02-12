import my_utils
import numpy as np
import matplotlib.pyplot as plt


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


def decision_boundary_plot(x, y):
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Scatter Plot and decision boundary')

    xone = x[:, 0]
    xtwo = x[:, 1]

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

    # x1_line = np.linspace(x1_lim[0], x1_lim[1], 200)
    # x1_line.shape = [200, 1]
    # theta = np.zeros([x.shape[1], ])
    # theta[2] = 1
    # x2_line = np.array([-((theta[0] + theta[1] * x1) / theta[2]) for x1 in x1_line])

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim(x1_lim)
    plt.ylim(x2_lim)

    y0 = plt.scatter(xone_yis0, xtwo_yis0, marker='o')
    y1 = plt.scatter(xone_yis1, xtwo_yis1, marker='x')
    # decision_boundary, = plt.plot(x1_line, x2_line, '#FF4500')
    ax.legend([y0, y1], ['Alaska', 'Canada'])
    # return decision_boundary,ax, y0, y1
    return ax, y0, y1


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

fig = plt.figure()

mng = plt.get_current_fig_manager()
# mng.full_screen_toggle()
mng.resize(*mng.window.maxsize())

decision_boundary_plot(X, Y)
mu0, mu1 = get_mu0(), get_mu1()
sigma0, sigma1 = get_covariance(mu0, mu1, same=False)
print(sigma0, "\n", sigma1)
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

'''
    Reading acidities

    X - list of acidities
'''
X = pd.read_csv("dataset/linearX.csv", header=None)
X = X.as_matrix()
X = np.insert(X, 0, 1.0, axis=1)  # x0 =

'''
    Reading densities
    Y - list of densities
'''
Y = pd.read_csv("dataset/linearY.csv", header=None)
Y = Y.as_matrix().flatten()


fig, ax = plt.subplots()
scatter = plt.scatter(list(np.delete(X, 0, axis=1).flatten()), list(Y.flatten()),c="red")
ln, = plt.plot([], [], animated=True)

def bgd(x, y, eeta, max_iter, threshold):
    num_examples = x.shape[0]
    num_features = x.shape[1] - 1

    # initializing thetas to all zeroes
    theta = np.zeros([num_features+1,])
    old_theta = np.zeros([num_features+1,])

    iter = 0
    while True:
        iter += 1

        for jth_feature in range(0, num_features + 1):
            # summation over i { y(i) - h theta x(i))* x(i)(j) }
            error = 0.0
            for ith_example in range(0, num_examples):
                error += (y[ith_example] - np.matmul(np.transpose(theta), x[ith_example]).flatten()[0]) * x[ith_example][jth_feature]
            theta[jth_feature] += eeta * error
        
        # convergance difference between values of theta
        difference = np.absolute(np.subtract(theta, old_theta))
        # print(iter, theta, difference[np.argmax(difference)])
        
        if (difference[np.argmax(difference)] < threshold or iter == max_iter):
            break

        old_theta = np.array(theta)
        if(iter % 100 == 0):
            yield theta, x, iter

def init():
    ax.set_xlim(5, 16)
    ax.set_ylim(0.975, 1.01)
    return ln,

def update(frame):
    theta, x, iter = frame
    x = np.delete(x, 0, axis=1)
    x = list(x.flatten())
    y_line = np.matmul(theta, np.transpose(X))
    y_line = list(y_line.flatten())
    ln.set_data(x, y_line)
    ax.legend(["iteration: "+str(iter)])
    return ax, ln

ani = FuncAnimation(fig, update, frames=bgd(X, Y, 0.0001,15000, 0.000001),
                    init_func=init, blit=True)
mng = plt.get_current_fig_manager()
mng.resize(1920, 1080)
plt.show()
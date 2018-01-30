import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

def plot(x, y, theta):
    y_line = np.matmul(theta, np.transpose(x))
    
    x = np.delete(x, 0, axis=1)
    x = list(x.flatten())
    y = list(y.flatten())
    y_line = list(y_line.flatten())
    plt.scatter(x,y)
    plt.plot(x,y_line)
    plt.xlabel("Acidity")
    plt.ylabel("Density")
    plt.ylim((0.975, 1.01))
    plt.draw()

def bgd(x, y, eeta, max_iter, threshold):
    num_examples = x.shape[0]
    num_features = x.shape[1] - 1

    # initializing thetas to all zeroes
    theta = np.zeros([num_features+1,])
    old_theta = np.zeros([num_features+1,])
    plot(x, y, theta)
    iter = 0
    while True:
        iter += 1

        for jth_feature in range(0, num_features + 1):
            # summation over i { y(i) - h theta x(i))* x(i)(j) }
            error = 0.0
            for ith_example in range(0, num_examples):
                error += (y[ith_example] - theta @ x[ith_example]).flatten()[0] * x[ith_example][jth_feature]
            theta[jth_feature] += eeta * error

        if(iter % 100 == 0):
            plt.gcf().clear()
            plot(x, y, theta)
            plt.legend(['iteration: '+str(iter)])
            plt.pause(0.2)
        
        # convergance difference between values of theta
        difference = np.absolute(np.subtract(theta, old_theta))
        print(iter, theta, difference[np.argmax(difference)])
        
        if (difference[np.argmax(difference)] < threshold or iter == max_iter):
            break

        old_theta = np.array(theta)

    plt.gcf().clear()
    plot(x, y, theta)
    plt.legend(['iteration: '+str(iter)])
    plt.show()

bgd(X, Y, 0.0001,15000, 0.000001)
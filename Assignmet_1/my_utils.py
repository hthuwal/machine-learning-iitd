import numpy as np
import pandas as pd

def normalize(x):
    mean = np.mean(x, axis=0)
    std = np.std(x)
    return np.apply_along_axis(lambda x: (x - mean) / std, 1, x)

def read_files(x_file, y_file):
	X = pd.read_csv("dataset/"+x_file, header=None)
	X = X.as_matrix()
	X = normalize(X)
	temp = X.flatten()
	std = np.std(temp)
	xlim = (temp[np.argmin(temp)] - std, temp[np.argmax(temp)] + std)
	X = np.insert(X, 0, 1.0, axis=1)


	Y = pd.read_csv("dataset/"+y_file, header=None)
	Y = Y.as_matrix().flatten()
	std = np.std(Y)
	ylim = (Y[np.argmin(Y)] - std, Y[np.argmax(Y)] + std)

	return X, Y, xlim, ylim
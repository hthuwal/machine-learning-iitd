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
Y = Y.as_matrix()

print(X)
print(Y)


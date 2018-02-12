import my_utils
import numpy as np

data = my_utils.read_files("q4x.dat", "q4y.dat", sep='\s+')
X = data[0]
Y = data[1]
cls_labels = data[4]
std = np.std(X, axis=0)
argmax = np.argmax(X, axis=0)
argmin = np.argmin(X, axis=0)

x1_lim = (X[argmin[1]][1] - std[1], X[argmax[1]][1] + std[1])
x2_lim = (X[argmin[2]][2] - std[2], X[argmax[2]][2] + std[2])
num_yi_is_1 = np.sum(Y) # because rest are zero so sum
num_yi_is_0 = len(Y) - num_yi_is_1
print(num_yi_is_0, num_yi_is_1)

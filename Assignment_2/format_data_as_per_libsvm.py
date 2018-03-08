import sys
import os
import pandas as pd
import numpy as np

def read_data(file):
    x = pd.read_csv(file, header=None)
    y = np.array(x[[784]]).flatten()
    x = x.drop(columns=[784])
    x = x.as_matrix()
    return x, y


def format(file):
	X, Y = read_data(file)
	basename, ext = os.path.splitext(file)
	with open(basename+"libsvm"+ext, "w") as f:
		for x, y in zip(X,Y):
			x = ["%d:%d" %(i+1, x[i]) for i in range(0, len(x))]
			x = " ".join(x)
			f.write("%d %s\n" %(y, x))

print("Formatting train file for libsvm")
format("../mnist/train.csv")
print("Formatting test file for libsvm")
format("../mnist/test.csv")
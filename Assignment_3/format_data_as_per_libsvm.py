import sys
import os
import pandas as pd
import numpy as np


def read_data(file):
    x = pd.read_csv(file, header=None)
    if x.shape[1] > 784:
        y = np.array(x[[784]]).flatten()
        x = x.drop(columns=[784])
    else:
        y = np.zeros(x.shape[0])
    x = x.as_matrix()
    return x, y


def format(file, formated_file):
    X, Y = read_data(file)
    basename, ext = os.path.splitext(file)
    with open(formated_file, "w") as f:
        for x, y in zip(X, Y):
            x = ["%d:%d" % (i + 1, x[i]) for i in range(0, len(x))]
            x = " ".join(x)
            f.write("%d %s\n" % (y, x))


file = sys.argv[1].strip()
output_file = sys.argv[2].strip()

print("Formatting %s in to format of libsvm" % file)
format(file, output_file)

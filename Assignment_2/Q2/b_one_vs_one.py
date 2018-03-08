from a_pegasos import bgd_pegasos
import numpy as np
import pandas as pd
import pickle
import os


def read_data(file):
    x = pd.read_csv(file, header=None)
    y = np.array(x[[784]]).flatten()
    x = x.drop(columns=[784])
    x = x.as_matrix()
    return x, y


x_train, y_train = read_data("../mnist/train.csv")
# x_test, y_test = read_data("../mnist/test.csv")

num_classes = len(set(y_train))

# training each classifier
retrain = False
wandbs = None
if retrain:
    wandbs = [[() for j in range(num_classes)] for i in range(num_classes)]
    count = 0
    for i in range(num_classes):
        for j in range(num_classes):
            if(i < j):
                count += 1
                print("\nClassifier %d: %d -> 1, %d -> -1\n" % (count, i, j))
                xc, yc = [], []
                for x, y in zip(x_train, y_train):
                    if (y == i):
                        xc.append(x)
                        yc.append(1)
                    elif(y == j):
                        xc.append(x)
                        yc.append(-1)

                wandbs[i][j] = bgd_pegasos(xc, yc, 10e-5)
    with open("model.pickle", "wb") as f:
        pickle.dump(wandbs, f)

else:
    with open("model.pickle", "rb") as f:
        wandbs = pickle.load(f)


def hypothesis(w, b, x):
    if (w@x + b) >= 0:
        return 1
    return -1


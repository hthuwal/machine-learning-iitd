from pegasos import bgd_pegasos

import numpy as np
import pandas as pd
import pickle
import sys


def read_data(file):
    x = pd.read_csv(file, header=None)
    if x.shape[1] > 784:
        y = np.array(x[[784]]).flatten()
        x = x.drop(columns=[784])
    else:
        y = np.zeros(x.shape[0])
    x = x.as_matrix()
    return x, y


def read_data_svm(file):
    x = []
    y = []
    with open(file, "r") as f:
        for line in f:
            temp = [0 for i in range(784)]
            line = line.strip().split(" ")
            y.append(int(line[0].strip()))
            line = line[1:]
            for each in line:
                each = each.split(":")
                temp[int(each[0].strip()) - 1] = np.float64(each[1].strip())

            x.append(temp)
            # input()
    x = np.array(x)
    y = np.array(y)
    # print(y.shape)
    return x, y


retrain = False
wandbs = None
if retrain:
    x_train, y_train = read_data("mnist/train.csv")
    num_classes = len(set(y_train))
    wandbs = [[() for j in range(num_classes)] for i in range(num_classes)]
    count = 0
    for i in range(num_classes):
        for j in range(num_classes):
            if(i < j):
                count += 1
                print("\nClassifier %d: %d vs %d\n" % (count, i, j))
                xc, yc = [], []
                for x, y in zip(x_train, y_train):
                    if (y == i):
                        xc.append(x)
                        yc.append(1)
                    elif(y == j):
                        xc.append(x)
                        yc.append(-1)

                wandbs[i][j] = bgd_pegasos(xc, yc, 10e-4, c=1.0)
    with open("models/pegasos.model", "wb") as f:
        pickle.dump(wandbs, f)

else:
    print("\nLoading Model")
    with open("models/pegasos.model", "rb") as f:
        wandbs = pickle.load(f)


def hypothesis(w, b, x):
    if (w@x + b) <= 0:
        return -1
    return 1


def predict(model, x):
    num_classes = len(model)
    counts = [0 for i in range(num_classes)]
    for i in range(num_classes):
        for j in range(num_classes):
            if(i < j):
                if hypothesis(model[i][j][0], model[i][j][1], x) == 1:
                    counts[i] += 1
                else:
                    counts[j] += 1
    return np.argmax(counts)


def run(x_set, y_set, model, output_file):
    correct = 0
    with open(output_file, "w") as f:
        for x, y in zip(x_set, y_set):
            prediction = predict(model, x)
            f.write("%d\n" % (prediction))
            if prediction == y:
                correct += 1

        accuracy = correct / (x_set.shape[0])
        print("Accuracy: %f\n" % (accuracy))


def run2(x_set, y_set, model, output_file):
    with open(output_file, "w") as f:
        length = len(x_set)
        for i in range(length):
            sys.stdout.write("\r\x1b[K" + "%d/%d : %0.2f percent" % (i + 1, length, (i + 1) * 100 / length))
            sys.stdout.flush()
            x, y = x_set[i], y_set[i]
            prediction = predict(model, x)
            f.write("%d\n" % (prediction))
    print("\n")


input_file = sys.argv[1].strip()
output_file = sys.argv[2].strip()
x_set, y_set = read_data(input_file)
print("Predicting")
run2(x_set, y_set, wandbs, output_file)

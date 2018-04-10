import numpy as np
import os
import pickle
from neural_network import Neural_Network
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


def read_data(x_file, y_file):
    data = []
    labels = []
    with open(x_file, "r") as x_file, open(y_file, "r") as y_file:
        for x, y in zip(x_file, y_file):
            x = x.strip().split(",")
            x = [float(each.strip()) for each in x]
            data.append(x)
            labels.append(int(y.strip()))
    return data, labels


train_data, train_labels = read_data("dataset/toy_data/toy_trainX.csv", "dataset/toy_data/toy_trainY.csv")
test_data, test_labels = read_data("dataset/toy_data/toy_testX.csv", "dataset/toy_data/toy_testY.csv")

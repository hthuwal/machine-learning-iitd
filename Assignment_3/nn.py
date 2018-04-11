import numpy as np
import pandas as pd
from neural_network import Neural_Network
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from visualization import plot_decision_boundary


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


def read_mnist(file):
    x = pd.read_csv(file, header=None)
    if x.shape[1] > 784:
        y = np.array(x[[784]]).flatten()
        x = x.drop(columns=[784])
    else:
        y = np.zeros(x.shape[0])
    x = np.array(x.as_matrix())
    y[y == 6] = 0
    y[y == 8] = 1
    return x, y


def b_1(plot=False):
    print("\nLogistic Regression")
    model = LogisticRegression()
    model.fit(train_data, train_labels)
    pred = model.predict(train_data)
    train_acc = accuracy_score(train_labels, pred) * 100
    print("Train Set Accuracy: ", train_acc)

    pred = model.predict(test_data)
    test_acc = accuracy_score(test_labels, pred) * 100
    print("Test Set Accuracy: ", test_acc)
    if plot:
        plot_decision_boundary(model.predict, np.array(train_data), np.array(train_labels), "LogisticRegression Train Set\n Accuracy: %f" % (train_acc))
        plot_decision_boundary(model.predict, np.array(test_data), np.array(test_labels), "LogisticRegression Test Set\n Accuracy: %f" % (test_acc))


def b_2(plot=False, units=[5], eeta=0.1, threshold=1e-6):
    print("\nNeural_Network")
    model = Neural_Network(len(train_data[0]), units, activation="sigmoid")
    print(model)
    model.train(train_data, train_labels, max_iter=5000, eeta=eeta, batch_size=len(train_data), threshold=threshold, decay=False)
    pred = model.predict(train_data)
    train_acc = accuracy_score(train_labels, pred) * 100
    print("Train Set Accuracy: ", train_acc)

    pred = model.predict(test_data)
    test_acc = accuracy_score(test_labels, pred) * 100
    print("Test Set Accuracy: ", test_acc)
    if plot:
        plot_decision_boundary(model.predict, np.array(train_data), np.array(train_labels), "Neural_Network Train Set\n Units in Hidden layers: %s\nAccuracy: %f" % (str(model.hidden_layer_sizes), train_acc))
        plot_decision_boundary(model.predict, np.array(test_data), np.array(test_labels), "Neural_Network Test Set\n Units in Hidden layers: %s\nAccuracy: %f" % (str(model.hidden_layer_sizes), test_acc))


def b_3(plot=False):
    units = [1, 2, 3, 10, 20, 40]
    lrs = [0.09, 0.09, 0.1, 0.1, 0.1, 0.01]
    # lrs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    for unit, lr in zip(units, lrs):
        print("\nNeural_Network")
        model = Neural_Network(len(train_data[0]), [unit], activation="sigmoid")
        print(model)
        model.train(train_data, train_labels, max_iter=10000, eeta=lr, batch_size=len(train_data), threshold=1e-6, decay=False)
        pred = model.predict(train_data)
        train_acc = accuracy_score(train_labels, pred) * 100
        print("Train Set Accuracy: ", train_acc)

        pred = model.predict(test_data)
        test_acc = accuracy_score(test_labels, pred) * 100
        print("Test Set Accuracy: ", test_acc)
        if plot:
            plot_decision_boundary(model.predict, np.array(test_data), np.array(test_labels), "Neural_Network Test Set\n Units in Hidden layers: %s\nAccuracy: %f" % (str(model.hidden_layer_sizes), test_acc))


def c_1(units=[]):
    print("\nNeural_Network MNIST")
    model = Neural_Network(len(mnist_trd[0]), units, activation="sigmoid")
    print(model)
    model.train(mnist_trd, mnist_trl, max_iter=250, eeta=0.001, batch_size=100, decay=True, threshold=1e-3)
    pred = model.predict(mnist_trd)
    train_acc = accuracy_score(mnist_trl, pred) * 100
    print("Train Set Accuracy: ", train_acc)

    pred = model.predict(mnist_ted)
    test_acc = accuracy_score(mnist_tel, pred) * 100
    print("Test Set Accuracy: ", test_acc)


def c_2(plot=False, units=[100], activation="sigmoid", eeta=0.1):
    print("\nNeural_Network MNIST")
    model = Neural_Network(len(mnist_trd[0]), units, activation=activation)
    print(model)
    model.train(mnist_trd, mnist_trl, max_iter=300, eeta=eeta, batch_size=100, decay=True, threshold=1e-3)
    pred = model.predict(mnist_trd)
    train_acc = accuracy_score(mnist_trl, pred) * 100
    print("Train Set Accuracy: ", train_acc)

    pred = model.predict(mnist_ted)
    test_acc = accuracy_score(mnist_tel, pred) * 100
    print("Test Set Accuracy: ", test_acc)


train_data, train_labels = read_data("dataset/toy_data/toy_trainX.csv", "dataset/toy_data/toy_trainY.csv")
test_data, test_labels = read_data("dataset/toy_data/toy_testX.csv", "dataset/toy_data/toy_testY.csv")

mnist_trd, mnist_trl = read_mnist("dataset/mnist_data/MNIST_train.csv")
mnist_ted, mnist_tel = read_mnist("dataset/mnist_data/MNIST_test.csv")
print(mnist_trl)
# b_1(plot=True)
# b_2(plot=True)
# b_3(plot=True)
b_2(plot=True, units=[5, 5], eeta=0.1, threshold=1e-10)
# c_1()
# c_2()
# c_2(plot=True, activation="ReLU", eeta=0.01)

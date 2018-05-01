import pickle
import numpy as np
from collections import Counter
import os
import sys
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

all_Acc = [
    35.171000,
    35.967000,
    35.560000,
    33.009000,
    35.205000,
    36.626000,
    35.782000,
    35.471000,
    36.812000,
    33.572000,
    34.358000,
    34.870000,
    36.442000,
    35.561000,
    35.066000,
    34.786000,
    35.450000,
    35.353000,
    35.039000,
    34.226000,
    31.910000,
    35.546000,
    33.037000,
    34.439000,
    34.004000,
    35.814000,
    35.197000,
    35.692000,
    35.622000
]


def all_plot():
    a, = plt.plot([10 * i for i in range(1, len(all_Acc) + 1)], all_Acc, 'ro', linestyle='solid')
    plt.xlabel("max_iter")
    plt.ylabel("Training Accuracy")
    plt.title("Variation in accuracy with change in max_iterations")
    plt.legend([a], ["Training Accuracy"])


def my_plot():
    x = [10, 20, 30, 40, 50]
    train_acc = [35.171, 35.967, 35.56, 33.009, 35.205]  # values obtained after running main
    test_acc = [34.930, 35.825, 34.862, 32.787, 35.140]  # values obtained from kaggle
    a, = plt.plot(x, train_acc, 'bo', linestyle='solid')
    b, = plt.plot(x, test_acc, 'go', linestyle='solid')
    # c, = plt.plot(x, test, 'r', linestyle='solid')
    plt.xlabel("max_iter")
    plt.ylabel("Accuracy")
    plt.title("Variation in accuracy with change in max_iterations")
    plt.legend([a, b], ["Training Accuracy", "Test Accuracy"])


def load_data(folder):
    labels = []
    data = None
    files = os.listdir(folder)
    for file in files:
        base, extension = os.path.splitext(file)
        if extension == ".npy":
            temp = np.load(os.path.join(folder, file))
        labels += [base] * (temp.shape[0])

        if data is None:
            data = np.array(temp)
        else:
            data = np.append(data, temp, axis=0)

    return data, np.array(labels)


def get_clustering_accuracy(labels):
    cluster_labels = ["" for i in range(20)]
    acc = 0
    for c in range(20):
        cluster_mem_lables = train_labels[np.where(labels == c)]
        most_common = Counter(cluster_mem_lables).most_common(1)[0]
        cluster_labels[c] = most_common[0]
        acc += (most_common[1])

    return (acc / len(labels) * 100, cluster_labels)


def predict(model, x, cluster_labels):
    labels = model.predict(x)
    ans = []
    for label in labels:
        ans.append(cluster_labels[label])
    return ans


def part3(file="kmeans%d.model", retrain=False, max_iter=300):

    old_file = "kmeans10.model"
    for i in range(10, 60, 10):
        file_name = file % (i)
        if os.path.exists(old_file):
            print("Loading %s" % (old_file))
            model = pickle.load(open(old_file, "rb"))
        else:
            model = KMeans(n_init=10, n_clusters=20, n_jobs=-1, verbose=0, max_iter=max_iter)

        if retrain:
            model.max_iter = max_iter
            print("Training Model: %d" % (i))
            model.fit(train_data)
            pickle.dump(model, open(file_name, "wb"))

        old_file = file_name
        train_acc, cluster_labels = get_clustering_accuracy(model.labels_)
        print("%d: Training Accuracy: %f" % (i, train_acc))
        predictions = predict(model, test_data, cluster_labels)
        with open("out%d.txt" % (i), "w") as f:
            f.write("ID,CATEGORY\n")
            for i in range(len(predictions)):
                f.write("%d,%s\n" % (i, predictions[i]))


def main2(file=None, retrain=False, max_iter=300):
    if os.path.exists(file):
        print("Loading %s..." % (file))
        model = pickle.load(open(file, "rb"))
    else:
        model = KMeans(n_init=10, n_clusters=20, n_jobs=1, verbose=1, max_iter=max_iter)

    if retrain:
        model.max_iter = max_iter
        print("Training Model...")
        model.fit(train_data)
        pickle.dump(model, open(file, "wb"))
        print("Training Complete...")

    print("Getting Training accuracy and cluster Labels..")
    train_acc, cluster_labels = get_clustering_accuracy(model.labels_)
    print("Training Accuracy: %f" % (train_acc))

    print("Predicting Test Data")
    predictions = predict(model, test_data, cluster_labels)
    with open("out.txt", "w") as f:
        f.write("ID,CATEGORY\n")
        for i in range(len(predictions)):
            f.write("%d,%s\n" % (i, predictions[i]))


if __name__ == '__main__':
    if sys.argv[1] == '-t':
        train_folder = sys.argv[2]
        model_file = sys.argv[3]
        print("Loading Train data...")
        train_data, train_labels = load_data(train_folder)
        model = KMeans(n_init=10, n_clusters=20, n_jobs=1, verbose=1, max_iter=1)
        print("Training Model...")
        model.fit(train_data)
        print("Training Complete...")
        print("Getting Training accuracy and cluster Labels..")
        train_acc, cluster_labels = get_clustering_accuracy(model.labels_)
        print("Training Accuracy: %f" % (train_acc))
        print("Dumping Model...")
        pickle.dump((model, cluster_labels), open(model_file, "wb"))

    elif sys.argv[1] == '-p':
        test_folder = sys.argv[2]
        model_file = sys.argv[3]
        print("Loading Model %s..." % (model_file))
        model, cluster_labels = pickle.load(open(model_file, "rb"))

        print("Loadint Test data")
        test_data, test_labels = load_data(test_folder)

        print("Predicting Test Data")
        predictions = predict(model, test_data, cluster_labels)
        # print(predictions)
        with open("out.txt", "w") as f:
            f.write("ID,CATEGORY\n")
            for i in range(len(predictions)):
                f.write("%d,%s\n" % (i, predictions[i]))
    else:
        print("Please Enter a valid parameter")

    # test_data, test_labels = load_data("dataset/test")
    # train_data, train_labels = load_data("dataset/train")
    # # part3(retrain=True, max_iter=100)
    # main2("temp.model", retrain=True, max_iter=1)
    # my_plot()
    all_plot()
    plt.show()

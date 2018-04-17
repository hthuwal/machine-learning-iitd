import pickle
import numpy as np
from collections import Counter
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


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


def main(file="models/kmeans%d.model", retrain=False, max_iter=300):

    old_file = "models/kmeans10.model"
    for i in range(10, 300, 10):
        file_name = file % (i)
        if os.path.exists(old_file):
            print("Loading %s" % (old_file))
            model = pickle.load(open(old_file, "rb"))
        else:
            model = KMeans(n_init=10, n_clusters=20, n_jobs=8, verbose=0, max_iter=max_iter)

        if retrain:
            model.max_iter = max_iter
            print("Training Model: %d" % (i))
            model.fit(train_data)
            pickle.dump(model, open(file_name, "wb"))

        old_file = file_name
        train_acc, cluster_labels = get_clustering_accuracy(model.labels_)
        print("%d: Training Accuracy: %f" % (i, train_acc))
        predictions = predict(model, test_data, cluster_labels)
        with open("outputs/out%d.txt" % (i), "w") as f:
            f.write("ID,CATEGORY\n")
            for i in range(len(predictions)):
                f.write("%d,%s\n" % (i, predictions[i]))


def main2(file, retrain=False, max_iter=300):
    if os.path.exists(file):
        print("Loading %s" % (file))
        model = pickle.load(open(file, "rb"))
    else:
        model = KMeans(n_init=10, n_clusters=20, n_jobs=8, verbose=1, max_iter=max_iter)

    if retrain:
        model.max_iter = max_iter
        print("Training Model")
        model.fit(train_data)
        pickle.dump(model, open(file, "wb"))

    train_acc, cluster_labels = get_clustering_accuracy(model.labels_)
    print("Training Accuracy: %f" % (train_acc))
    predictions = predict(model, test_data, cluster_labels)
    with open("out.txt", "w") as f:
        f.write("ID,CATEGORY\n")
        for i in range(len(predictions)):
            f.write("%d,%s\n" % (i, predictions[i]))


train_data, train_labels = load_data("dataset/train")
test_data, test_labels = load_data("dataset/test")
# main(retrain=True, max_iter=10)
# main2("cum_300.model", retrain=True)
my_plot()
plt.show()

# Multinomial Event Model
# Given a review predict the rating (1-10)
# y is Multinomial phi1 to phi10
# Every position has same multinomial theta1 to theta|V|

import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import re
import sys
import random
from collections import Counter
from tqdm import tqdm
import pickle
# TODO code should work even if tqdm is absent


def clean(string):
    string = string.lower().strip()
    string = re.sub("[^a-z0-9]", " ", string)  # removing all accept letters and numbers
    return string.split()


def read_data(review_file, rating_file):
    print("Reading Files \'%s\' and \'%s\'\n" % (review_file, rating_file))
    data = []
    with open(review_file, 'r') as rev, open(rating_file, 'r') as rt:
        for review, rating in zip(rev, rt):
            rating = int(rating)
            review = clean(review)  # clean review and return a list of words
            data.append((rating, review))
    return data


def format_data(plain_data):
    data = {}
    for rating, review in plain_data:
        if rating not in data:
            data[rating] = {"words": list(review), "num_of_samples": 1}
        else:
            data[rating]["words"] += review
            data[rating]["num_of_samples"] += 1

    for rating in data:
        data[rating]["num_of_words"] = len(data[rating]["words"])
        data[rating]["words"] = Counter(data[rating]["words"])

    return data


def get_vocab(data):
    v = Counter([])
    for rating in data:
        v += data[rating]["words"]
    return v


phis = {}
thetas = {}

notlist = {
    1: 3,
    2: 4,
    3: 7,
    4: 7,
    7: 2,
    8: 1,
    9: 1,
    10: 1
}

negations = ["neiter", "nor", "nothing", "didnt", "not", "never", "nope", "none", "no", "nobody", "noway", "nah"]


def predict(review, c):
    probs = [0 for i in range(0, num_classes)]
    # probs = np.zeros([num_classes, ])
    classes = list(data.keys())

    probs = dict(zip(classes, probs))

    for cls in probs:
        # log(phi_cls)
        if cls not in phis:
            phis[cls] = math.log10((data[cls]["num_of_samples"] + c) / (total_num_of_samples + c * num_classes))
        probs[cls] += phis[cls]

        if cls not in thetas:
            thetas[cls] = {}
        for i, word in enumerate(review):
            # log(theta_word_cls)
            if word not in thetas[cls]:
                thetas[cls][word] = math.log10((data[cls]["words"][word] + c) / (data[cls]["num_of_words"] + c * V))
            if (i>=1 and review[i-1] in negations) or (i>=2 and review[i-2] in negations):
                probs[cls] -= (thetas[cls][word])
            else:
                probs[cls] += thetas[cls][word]

    keys = list(probs.keys())
    max_cls = keys[0]

    for cls in probs:
        if probs[cls] > probs[max_cls]:
            max_cls = cls

    return max_cls


def run(dataset, method='naive_bayes', confusion=False):
    count = 0
    num_samples = len(dataset)
    correct_prediction = 0

    for actual_cls, review in tqdm(dataset):
        count += 1
        # print(count)
        if method == "naive_bayes":
            prediction = predict(review, 1)
            if actual_cls == prediction:
                correct_prediction += 1

            if confusion:
                if prediction > 4:
                    prediction -= 2
                if actual_cls > 4:
                    actual_cls -= 2
                cf_mat[actual_cls - 1][prediction - 1] += 1

        elif method == "random":
            if actual_cls == random_prediction():
                correct_prediction += 1

        elif method == "maxcls":
            if actual_cls == maxcls:
                correct_prediction += 1

    return (correct_prediction / num_samples) * 100


def random_prediction():
    classes = list(data.keys())
    i = random.randint(0, 7)
    return classes[i]


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '0.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if len(sys.argv) == 2 and sys.argv[1] == "stemmed":
    training_data = read_data("imdb/imdb_train_text_stemmed.txt", "imdb/imdb_train_labels.txt")
    testing_data = read_data("imdb/imdb_test_text_stemmed.txt", "imdb/imdb_test_labels.txt")
    output_file = "models/naive_bayes_stemmed.model"
else:
    training_data = read_data("imdb/imdb_train_text.txt", "imdb/imdb_train_labels.txt")
    testing_data = read_data("imdb/imdb_test_text.txt", "imdb/imdb_test_labels.txt")
    output_file = "models/naive_bayes.model"

data = format_data(training_data)
num_classes = len(data)
vocab = get_vocab(data)
V = len(vocab)
total_num_of_samples = 0
for rating in data:
    total_num_of_samples += data[rating]["num_of_samples"]

cf_mat = np.zeros([8, 8])  # confusion_matrix

print("Running on Training data")
train_accuracy = run(training_data)
print("Training Accuracy: %f\n" % (train_accuracy))


print("Running on Testing data")
test_accuracy = run(testing_data, confusion=True)
print("Test Accuracy: %f\n" % (test_accuracy))


print("Random Prediction on Test Set")
test_accuracy = run(testing_data, method="random")
print("Accuracy: %f\n" % (test_accuracy))

print("Majority Prediction on Test Set")
maxcls = list(data.keys())[0]
for cls in data:
    if data[cls]["num_of_samples"] > data[maxcls]["num_of_samples"]:
        maxcls = cls

test_accuracy = run(testing_data, method="maxcls")
print("Accuracy: %f\n" % (test_accuracy))

# Confusion Matrix
classes = list(data.keys())
classes.sort()
plt.figure()
plot_confusion_matrix(cf_mat, classes=classes, title='Confusion matrix')  # , cmap=plt.cm.viridis_r)
plt.show()

with open(output_file, "wb") as f:
    for cls in data:
        del data[cls]["words"]
    pickle.dump((phis, thetas, V, data), f)

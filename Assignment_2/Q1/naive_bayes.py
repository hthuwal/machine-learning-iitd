# Multinomial Event Model
# Given a review predict the rating (1-10)
# y is Multinomial phi1 to phi10
# Every position has same multinomial theta1 to theta|V|

import numpy as np
import re
from collections import Counter

extra = ["?", ".", "\"", "\'", "/", "\\", ":", ";", "(", ")"]


def clean(string):
    # TODO: this should remove faltu symbols
    string = string.lower().strip()
    string = re.sub("[^a-z0-9]", " ", string)  # removing all accept letters and numbers
    return string.split()


def read_data(review_file, rating_file):
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


def predict(review, c):
    probs = np.zeros([num_classes, ])
    classes = list(data.keys())

    probs = dict(zip(classes, probs))

    for cls in probs:
        # log(phi_cls)
        probs[cls] += np.log10((data[cls]["num_of_samples"] + c) / (total_num_of_reviews + c * num_classes))
        for word in review:
            # log(theta_word_cls)
            probs[cls] += np.log10((data[cls]["words"][word] + c) / (data[cls]["num_of_words"] + c * V))

    keys = list(probs.keys())
    max_cls = keys[0]

    for cls in probs:
        if probs[cls] > probs[max_cls]:
            max_cls = cls

    return max_cls


def run(dataset):
    count = 0
    num_samples = len(dataset)
    correct_prediction = 0
    for actual_cls, review in dataset:
        count += 1
        print(count)
        if(actual_cls == predict(review, 1)):
            correct_prediction += 1

    return (correct_prediction / num_samples) * 100


training_data = read_data("../imdb/imdb_train_text.txt", "../imdb/imdb_train_labels.txt")
testing_data = read_data("../imdb/imdb_test_text.txt", "../imdb/imdb_test_labels.txt")
data = format_data(training_data)
num_classes = len(data)
vocab = get_vocab(data)
V = len(vocab)
total_num_of_reviews = 0
for rating in data:
    total_num_of_reviews += data[rating]["num_of_samples"]


phis = dict(zip(data.keys(), np.zeros([num_classes, ])))
thetas = {}
for word in vocab:
    thetas[word] = dict(phis)


train_accuracy = run(training_data)
test_accuracy = run(testing_data)

print("Training Accuracy: %f\nTest Accuracy: %f\n" % (train_accuracy, test_accuracy))

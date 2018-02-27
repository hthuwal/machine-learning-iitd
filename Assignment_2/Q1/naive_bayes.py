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
            data[rating] = {"words": list(review), "num_of_reviews": 1}
        else:
            data[rating]["words"] += review
            data[rating]["num_of_reviews"] += 1

    for rating in data:
        data[rating]["num_of_words"] = len(data[rating]["words"])
        data[rating]["words"] = Counter(data[rating]["words"])

    return data


def get_vocab(data):
    v = Counter([])
    for rating in data:
        v += data[rating]["words"]
    return v


def calculate_paramters(data, c):
    total_num_of_reviews = 0
    for rating in data:
        total_num_of_reviews += data[rating]["num_of_reviews"]

    for rating in data:
        phis[rating] = (data[rating]["num_of_reviews"] + c) / (total_num_of_reviews + c * num_classes)

    for word in thetas:
        for cls in thetas[word]:
            thetas[word][cls] = (data[cls]["words"][word] + c) / (data[cls]["num_of_words"] + c * V)


training_data = read_data("../imdb/imdb_train_text.txt", "../imdb/imdb_train_labels.txt")
data = format_data(training_data)
num_classes = len(data)
vocab = get_vocab(data)
V = len(vocab)


phis = dict(zip(data.keys(), np.zeros([num_classes, ])))
thetas = {}
for word in vocab:
    thetas[word] = dict(phis)

calculate_paramters(data, 1)

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
    string = string.strip()
    string1 = re.sub("[^a-z0-9]", " ", string)  # removing all accept letters and numbers
    string2 = re.sub("[^a-z0-9]", "", string)  # removing all accept letters and numbers
    return list(set(string1.split() + string2.split()))


def read_data(review_file):
    print("Reading Files \'%s\'\n" % (review_file))
    data = []
    with open(review_file, 'r') as rev:
        for review in rev:
            review = clean(review)  # clean review and return a list of words
            data.append(review)
    return data


def predict(review, c, V, data):
    classes = list(data.keys())
    probs = [0 for i in range(0, len(classes))]
    # probs = np.zeros([num_classes, ])
    probs = dict(zip(classes, probs))

    review.sort()
    for cls in probs:
        # log(phi_cls)
        probs[cls] += phis[cls]

        for word in review:

            if word not in thetas[cls]:
                thetas[cls][word] = math.log10((0 + c) / (data[cls]["num_of_words"] + c * V))
            probs[cls] += thetas[cls][word]

    keys = list(probs.keys())
    max_cls = keys[0]

    for cls in probs:
        if probs[cls] > probs[max_cls]:
            max_cls = cls

    return max_cls


negations = ["neiter", "nor", "nothing", "didnt", "not", "never", "nope", "none", "no", "nobody", "noway", "nah", "aint"]


def predict2(review, c, V, data):
    classes = list(data.keys())
    probs = [0 for i in range(len(classes))]
    # probs = np.zeros([num_classes, ])
    classes = list(data.keys())

    probs = dict(zip(classes, probs))

    review.sort()
    for cls in probs:
        # log(phi_cls)
        probs[cls] += phis[cls]

        for i in range(len(review)):
            word = review[i]
            # log(theta_word_cls)
            if word not in thetas[cls]:
                thetas[cls][word] = math.log10((0 + c) / (data[cls]["num_of_words"] + c * V))
            if (i == 0 or i == 1 or i == 2):
                probs[cls] += 2 * thetas[cls][word]
            elif (i >= 1 and review[i - 1] in negations) or (i >= 2 and review[i - 2] in negations):
                probs[cls] -= (thetas[cls][word])
            else:
                probs[cls] += thetas[cls][word]

    keys = list(probs.keys())
    max_cls = keys[0]

    for cls in probs:
        if probs[cls] > probs[max_cls]:
            max_cls = cls

    return max_cls


def run(dataset, V, data, output_file, e=False):
    count = 0
    num_samples = len(dataset)
    correct_prediction = 0

    with open(output_file, "w") as f:
        for review in tqdm(dataset):
            if e:
                prediction = predict2(review, 1, V, data)
            else:
                prediction = predict(review, 1, V, data)
            f.write("%d\n" % prediction)


model = None
dataset = None
if sys.argv[1] == "1":
    print("Loading Model naive_bayes.model\n")
    model = pickle.load(open("models/naive_bayes.model", "rb"))
    dataset = read_data(sys.argv[2].strip())
elif sys.argv[1] == "2":
    print("Loading Model naive_bayes_stemmed.model\n")
    model = pickle.load(open("models/naive_bayes_stemmed.model", "rb"))
    dataset = read_data(sys.argv[2].strip())
elif sys.argv[1] == "3":
    print("Loading Model naive_bayes_stemmed_e.model\n")
    model = pickle.load(open("models/naive_bayes_stemmed_e.model", "rb"))
    dataset = read_data(sys.argv[2].strip())

phis = model[0]
thetas = model[1]
V = model[2]
data = model[3]
if sys.argv[1] == "3":
    run(dataset, V, data, sys.argv[3].strip(), e=True)
else:
    run(dataset, V, data, sys.argv[3].strip())

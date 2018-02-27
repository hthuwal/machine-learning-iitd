# Multinomial Event Model
# Given a review predict the rating (1-10)
# y is Multinomial phi1 to phi10
# Every position has same multinomial theta1 to theta|V|
import re
from collections import Counter

extra = ["?", ".", "\"", "\'", "/", "\\", ":", ";", "(", ")"]


def clean(string):
    # TODO: this should remove faltu symbols
    string = string.lower().strip()
    string = re.sub("[^a-z0-9]", " ", string)  # removing all accept letters and numbers
    return string.split()


def read_review_and_ratings(review_file, rating_file):

    data = {}
    with open(review_file, 'r') as rev, open(rating_file, 'r') as rt:
        for review, rating in zip(rev, rt):
            rating = int(rating)
            review = clean(review)  # clean review and return a list of words

            if rating not in data:
                data[rating] = review
            else:
                data[rating] += review

    for key in data:
        data[key] = (Counter(data[key]), len(data[key]))

    return data


def get_vocab(data):
    v = Counter([])
    for rating in data:
        v += data[rating][0]
    return v


data = read_review_and_ratings("../imdb/imdb_train_text.txt", "../imdb/imdb_train_labels.txt")
vocab = get_vocab(data)
V = len(vocab)

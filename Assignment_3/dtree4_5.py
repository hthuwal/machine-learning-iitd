import read_data as rd
import numpy as np
# from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
# from tqdm import tqdm
# import matplotlib.pyplot as plt

attributes = rd.attributes[1:]
attributes = [(i, attr) for i, attr in enumerate(attributes)]

train_data = rd.preprocess("dataset/dtree_data/train.csv", binarize=False)
train_labels = np.array(train_data[:, 0])
train_data = np.delete(train_data, 0, 1)
test_data = rd.preprocess("dataset/dtree_data/test.csv", binarize=False)
test_labels = np.array(test_data[:, 0])
test_data = np.delete(test_data, 0, 1)
valid_data = rd.preprocess("dataset/dtree_data/valid.csv", binarize=False)
valid_labels = np.array(valid_data[:, 0])
valid_data = np.delete(valid_data, 0, 1)


decison_tree = DecisionTreeClassifier()
# decison_tree.fit(train_data, train_labels)


# for depth in range(10,50,5):
#     for split in range(10, 1000, 100):
#         for leaf in range(10, 1000, 100):
#             decison_tree = DecisionTreeClassifier(criterion="entropy", max_depth=depth, min_samples_split=split, min_samples_leaf=leaf)
#             decison_tree.fit(train_data, train_labels)
#             print(accuracy_score(train_labels, decison_tree.predict(train_data)))
#             print(accuracy_score(valid_labels, decison_tree.predict(valid_data)))
#             print(accuracy_score(test_labels, decison_tree.predict(test_data)))
parameters = {
    'criterion': ("gini", "entropy"),
    'max_depth': (None, 10, 20, 30),
    'min_samples_split': (2, 200, 400, 600, 800, 1000),
    'min_samples_leaf': (1, 200, 400, 600, 800, 1000),
}

# model = GridSearchCV(decison_tree, parameters, verbose=10, n_jobs=2)
# model.fit(train_data, train_labels)

# best model after grid search
decison_tree = DecisionTreeClassifier(criterion="entropy", max_depth=10, min_samples_leaf=1, min_samples_split=2)
decison_tree.fit(train_data, train_labels)
print(accuracy_score(train_labels, decison_tree.predict(train_data)))
print(accuracy_score(valid_labels, decison_tree.predict(valid_data)))
print(accuracy_score(test_labels, decison_tree.predict(test_data)))

# print("\nTraining accuracy: %0.2f Validation accuracy: %0.2f Test accuracy: %0.2f" % accuracy(bfs_fast))

# part_c(step=10)
# pruning()

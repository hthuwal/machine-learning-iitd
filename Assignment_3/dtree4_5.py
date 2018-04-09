import read_data as rd
import numpy as np
# from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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

parameters_dtree = {
    'criterion': ("gini", "entropy"),
    'max_depth': (None, 10, 20, 30),
    'min_samples_split': (2, 200, 400, 600, 800, 1000),
    'min_samples_leaf': (1, 200, 400, 600, 800, 1000),
}
# decison_tree = DecisionTreeClassifier()
# model = GridSearchCV(decison_tree, parameters_dtree, verbose=10, n_jobs=2)
# model.fit(train_data, train_labels)

# best model after grid search
decison_tree = DecisionTreeClassifier(criterion="entropy", max_depth=10, min_samples_leaf=1, min_samples_split=2)
decison_tree.fit(train_data, train_labels)
print(accuracy_score(train_labels, decison_tree.predict(train_data)))
print(accuracy_score(valid_labels, decison_tree.predict(valid_data)))
print(accuracy_score(test_labels, decison_tree.predict(test_data)))

# parameters_rforest = {
#     'bootstrap': (True, False),
#     'max_features': (2, 5, 8, 10, 12, 14, 'auto', 'log2', None),
#     'n_estimators': (20, 25, 30,40, 50, 60),
# }

# random_forest = RandomForestClassifier(criterion="entropy", max_depth=10, min_samples_leaf=1, min_samples_split=2)
# model = GridSearchCV(random_forest, parameters_rforest, verbose=5, n_jobs=4)
# model.fit(train_data, train_labels)

# best model after grid search
print("RandomForestClassifier")
random_forest = RandomForestClassifier(criterion="entropy", max_depth=10, min_samples_leaf=1, min_samples_split=2, max_features=12, n_estimators=20, bootstrap=True)
random_forest.fit(train_data, train_labels)
print(accuracy_score(train_labels, random_forest.predict(train_data)))
print(accuracy_score(valid_labels, random_forest.predict(valid_data)))
print(accuracy_score(test_labels, random_forest.predict(test_data)))

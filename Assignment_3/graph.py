import read_data as rd
import numpy as np
# import pdb
from collections import Counter
from sklearn.metrics import accuracy_score

attributes = rd.attributes[1:]
attributes = [(i, attr) for i, attr in enumerate(attributes)]
nodes = []
graph = {}


class Node(object):
    def __init__(self, index):
        self.index = index
        self.majority = None
        self.di = None
        self.sa = None

    def __repr__(self):
        return str(self.index)


def entropy(labels):
    labels = Counter(labels)
    if (len(labels) == 2):  # only if two categories present
        prob1 = labels[0] / (labels[0] + labels[1])
        prob2 = labels[1] / (labels[0] + labels[1])
        return -(prob1 * np.log2(prob1) + prob2 * np.log2(prob2))

    return 0


def information_gain(data, labels, attr, attr_index):
    ebs = entropy(labels)  # entropy before splitting
    if attr in rd.encode:
        nv = len(rd.encode[attr])  # number of values this attribute can take
    else:
        nv = 2

    total_samples = len(labels)
    eas = 0
    ioc = []
    for value in range(nv):
        ias, = np.where(data[:, attr_index] == value)
        las = labels[ias]
        eas += (len(ias) / total_samples) * entropy(las)
        ioc.append(ias)

    return (ebs - eas), ioc


def best_attribute(data, labels, attributes):
    best_attr = (None, -1, None)
    for attr in attributes:
        igain, c = information_gain(data, labels, attr[1], attr[0])
        if igain > 0 and igain > best_attr[1]:
            best_attr = (attr, igain, c)
    return best_attr


def create_node(labels, indices):
    ltc = Counter(labels[indices])
    node = Node(len(nodes))
    node.di = indices
    node.majority = 0 if ltc[0] >= ltc[1] else 1
    nodes.append(node)
    return node


def is_alone(labels):
    if len(Counter(labels)) < 2:
        return True
    return False


def build_tree(data, labels, indices, attrib):
    print("\r\x1b[K" + str(len(nodes)), end=" ")
    dtc = data[indices]
    ltc = labels[indices]

    if len(indices) == 0:  # no data
        return None

    node = create_node(labels, indices)

    if len(attrib) == 0 or is_alone(ltc):  # no attrib to split on
        return node.index

    best_attr, igain, childs = best_attribute(dtc, ltc, attrib)
    if best_attr is None:
        return node.index

    node.sa = best_attr
    attrib.remove(best_attr)
    for split_attr_value, child_indices in enumerate(childs):
        child_index = build_tree(dtc, ltc, child_indices, list(attrib))
        if child_index is not None:
            if node.index not in graph:
                graph[node.index] = {}
            graph[node.index][split_attr_value] = child_index

    return node.index


ls = []


def dfs(root):
    ls.append(root)
    if root in graph:
        for key in graph[root]:
            dfs(graph[root][key])


def height(root):
    if root not in graph:
        return 0

    ans = 0
    for key in graph[root]:
        x = height(graph[root][key])
        if x > ans:
            ans = x
    return ans + 1


def predict(node_index, x):
    node = nodes[node_index]
    # print(node_index, node.sa)
    if node.sa is None:
        return node.majority

    key = x[node.sa[0]]
    if key not in graph[node.index]:
        return node.majority

    return predict(graph[node.index][key], x)


def get_accuracy(data, labels):
    pred = []
    for i, x in enumerate(data):
        pred.append(predict(0, x))
    return accuracy_score(labels, pred)


def accuracy():
    train_acc = get_accuracy(train_data, train_labels) * 100
    valid_acc = get_accuracy(valid_data, valid_labels) * 100
    test_acc = get_accuracy(test_data, test_labels) * 100

    return train_acc, valid_acc, test_acc


train_data = rd.preprocess("dataset/dtree_data/train.csv")
train_labels = np.array(train_data[:, 0])
train_data = np.delete(train_data, 0, 1)
test_data = rd.preprocess("dataset/dtree_data/test.csv")
test_labels = np.array(test_data[:, 0])
test_data = np.delete(test_data, 0, 1)
valid_data = rd.preprocess("dataset/dtree_data/valid.csv")
valid_labels = np.array(valid_data[:, 0])
valid_data = np.delete(valid_data, 0, 1)


# data = data[:10]
# labels = labels[:10]

build_tree(train_data, train_labels, [i for i in range(len(train_data))], list(attributes))
print("\nTraining accuracy: %0.2f Validation accuracy: %0.2f Test accuracy: %0.2f" % accuracy())

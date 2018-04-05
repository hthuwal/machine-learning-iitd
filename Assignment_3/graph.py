import read_data as rd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from tqdm import tqdm
graph = {}
nodes = []


class Node(object):
    def __init__(self, index):
        self.index = index
        self.majority_cls = -1
        self.split_attr = None
        self.data_indices = None


def log2(x):
    if x == 0:
        return 0
    else:
        return np.log2(x)


def entropy(labels, data_indices):
    temp = np.array(labels)[data_indices]
    temp = Counter(temp)

    if len(temp) == 0:
        return 0

    prob1 = temp[0] / (temp[0] + temp[1])
    prob2 = temp[1] / (temp[0] + temp[1])
    entropy = prob1 * (-log2(prob1)) + prob2 * (-log2(prob2))
    return entropy


def information_gain(data, labels, data_indices, attr, attr_index):
    if attr in rd.encode:
        pos_attr_values = len(rd.encode[attr])
    else:
        pos_attr_values = 2

    before_entropy = entropy(labels, data_indices)
    after_entropy = 0

    x = data[data_indices]
    total = len(x)

    children_indices = []
    for i in range(pos_attr_values):
        indices, = np.where(x[:, attr_index] == i)
        children_indices.append(indices)
        after_entropy += (entropy(labels, indices) * (len(indices) / total))

    information = before_entropy - after_entropy
    return (children_indices, information)


def select_best_attr(data, labels, data_indices, attribs):
    best_children = None
    best_info_gain = -1
    best_attr = None
    for i, attr in attribs:
        children, info_gain = information_gain(data, labels, data_indices, attr, i)
        if info_gain > 0 and info_gain > best_info_gain:
            best_children = children
            best_info_gain = info_gain
            best_attr = (i, attr)
    return best_attr, best_children


def create_node(labels, data_indices):
    node = Node(len(nodes))
    node.data_indices = data_indices
    temp = Counter(labels[data_indices])
    node.majority_cls = 0 if temp[0] > temp[1] else 1
    nodes.append(node)
    return node


def height(graph, index):
    if index not in graph:
        return 0
    elif len(graph[index]) == 0:
        return 0
    else:
        return 1 + max([height(graph, graph[index][j].index) for j in graph[index]])


def build_tree(graph, data, labels, data_indices, attribs):
    if len(Counter(labels[data_indices])) == 1:
        return create_node(labels, data_indices)

    elif len(data_indices) == 0:
        print("Yeh kaise ho gaya?")
        return None

    elif len(data_indices) == 1:
        node = Node(len(nodes))
        node.data_indices = data_indices
        node.majority_cls = labels[data_indices][0]
        nodes.append(node)
        return node

    elif len(attribs) == 0:
        return create_node(labels, data_indices)

    else:
        best_attr, best_children = select_best_attr(data, labels, data_indices, attribs)
        if best_attr is None:
            return create_node(labels, data_indices)
        else:
            node = create_node(labels, data_indices)
            node.split_attr = best_attr
            # attribs.remove(best_attr)
            best_children = [build_tree(graph, data, labels, children_indices, attribs)for children_indices in best_children]
            for i, child in enumerate(best_children):
                if node.index not in graph:
                    graph[node.index] = {}
                if child is not None:
                    graph[node.index][i] = child
            return node


def test(graph, root, x):
    if root.split_attr == None:
        return root.majority_cls

    attr_value = x[root.split_attr[0]]
    if attr_value in graph[root.index]:
        return test(graph, graph[root.index][attr_value], x)

    return root.majority_cls


def get_accuracy(graph, root, x, y):
    pred = []
    for each in tqdm(x):
        pred.append(test(graph, root, each))

    return accuracy_score(y, pred)


data = rd.preprocess("dataset/dtree_data/train.csv")
attributes = rd.attributes[1:]
attributes = [(i, attr) for i, attr in enumerate(attributes)]
labels = np.array(data[:, 0])
data = np.delete(data, 0, 1)

build_tree(graph, data, labels, [i for i in range(len(data))], attributes)

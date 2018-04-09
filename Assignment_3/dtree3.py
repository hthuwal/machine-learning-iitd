import read_data as rd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

attributes = rd.attributes[1:]
attributes = [(i, attr) for i, attr in enumerate(attributes)]
nodes = []
graph = {}


def my_plot(train, valid, test, x):
    a, = plt.plot(x, train, 'b', linestyle='solid')
    b, = plt.plot(x, valid, 'g', linestyle='solid')
    c, = plt.plot(x, test, 'r', linestyle='solid')
    plt.xlabel("Number of Nodes")
    plt.ylabel("Accuracies")
    plt.title("Number of Nodes: %d" % x[-1])
    plt.legend([a, b, c], ["Training Accuracy: %0.2f" % train[-1], "Validation Accuracy: %0.2f" % valid[-1], "Test Accuracy: %0.2f" % test[-1]])
    return fig


class Node(object):
    def __init__(self, index):
        self.index = index
        self.majority = None
        self.di = None
        self.sa = None
        self.median = None

    def __repr__(self):
        return str(self.index)


def entropy(labels):
    labels = Counter(labels)
    if (len(labels) == 2):  # only if two categories present
        prob1 = labels[0] / (labels[0] + labels[1])
        prob2 = labels[1] / (labels[0] + labels[1])
        return -(prob1 * np.log2(prob1) + prob2 * np.log2(prob2))

    return 0


def binarize(l, index, median):
    l[index] = l[index] >= median


def information_gain(data, labels, attr, attr_index):
    ebs = entropy(labels)  # entropy before splitting
    if attr in rd.encode:  # if attribute is categorical
        nv = len(rd.encode[attr])  # number of values this attribute can take
        total_samples = len(labels)
        eas = 0
        ioc = []
        for value in range(nv):
            ias, = np.where(data[:, attr_index] == value)
            las = labels[ias]
            eas += (len(ias) / total_samples) * entropy(las)
            ioc.append(ias)

        return (ebs - eas), ioc, None

    else:
        attr_value_list = data[:, attr_index]
        median = np.median(attr_value_list)
        new_data = np.array(data)
        np.apply_along_axis(binarize, 1, new_data, attr_index, median)
        nv = 2
        total_samples = len(labels)
        eas = 0
        ioc = []
        for value in range(nv):
            ias, = np.where(new_data[:, attr_index] == value)
            las = labels[ias]
            eas += (len(ias) / total_samples) * entropy(las)
            ioc.append(ias)

        return (ebs - eas), ioc, median


def best_attribute(data, labels, attributes):
    best_attr = (None, -1, None, None)
    for attr in attributes:
        igain, c, median = information_gain(data, labels, attr[1], attr[0])
        if igain > 0 and igain > best_attr[1]:
            best_attr = (attr, igain, c, median)
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

    best_attr, igain, childs, median = best_attribute(dtc, ltc, attrib)
    if best_attr is None:
        return node.index

    node.sa = best_attr
    node.median = median
    if best_attr in rd.encode:  # if best_attr is categorical then can't be used again
        attrib.remove(best_attr)
    for split_attr_value, child_indices in enumerate(childs):
        child_index = build_tree(dtc, ltc, child_indices, list(attrib))
        if child_index is not None:
            if node.index not in graph:
                graph[node.index] = {}
            graph[node.index][split_attr_value] = child_index

    return node.index


ls = []


def bfs(root):
    ans = []
    hc = []
    ans.append(root)
    hc.append(root)
    while(len(hc) != 0):
        front = hc[-1]
        hc.pop()
        if front in graph:
            for key in graph[front]:
                ans.append(graph[front][key])
                hc.insert(0, graph[front][key])
    return ans


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


def predict(node_index, x, bfs_fast):
    node = nodes[node_index]
    if bfs_fast[node_index] == 0:
        return node.majority

    if node.sa is None:
        return node.majority

    if node.sa[1] not in rd.encode:
        key = int(x[node.sa[0]] >= node.median)
    else:
        key = x[node.sa[0]]

    if key not in graph[node.index]:
        return node.majority

    if bfs_fast[graph[node.index][key]] == 0:
        return node.majority

    return predict(graph[node.index][key], x, bfs_fast)


def get_accuracy(data, labels, bfs_fast):
    pred = []
    for i, x in enumerate(data):
        pred.append(predict(0, x, bfs_fast))
    return accuracy_score(labels, pred)


def accuracy(bfs_fast):
    train_acc = get_accuracy(train_data, train_labels, bfs_fast) * 100
    valid_acc = get_accuracy(valid_data, valid_labels, bfs_fast) * 100
    test_acc = get_accuracy(test_data, test_labels, bfs_fast) * 100

    return train_acc, valid_acc, test_acc


fig = plt.figure()


def part_c(step=10):
    train = []
    test = []
    valid = []
    x = []

    for i in tqdm(range(0, len(nodes), step)):
        bfs_fast = np.zeros(len(nodes))

        for j in range(len(bfs_order[0:i])):
            bfs_fast[bfs_order[j]] = 1

        ta, va, tea = accuracy(bfs_fast)
        train.append(ta)
        valid.append(va)
        test.append(tea)
        x.append(i)

        # fig.clf()
        # my_plot(train, valid, test, x)
        # plt.pause(0.001)
        # print("\nTraining accuracy: %0.2f Validation accuracy: %0.2f Test accuracy: %0.2f" % (ta, va, tea))

    fig.clf()
    my_plot(train, valid, test, x)
    plt.pause(0.01)
    plt.show()


def find_num_nodes_in_tree(root_index):
    if root_index not in graph:
        return 1
    else:
        ans = 1
        for key in graph[root_index]:
            ans += find_num_nodes_in_tree(graph[root_index][key])
        return ans


def num_of_times_used_as_split(root_index, attr):
    if root_index not in graph:
        if nodes[root_index].sa is not None and nodes[root_index].sa[1] == attr:
            return [(root_index, nodes[root_index].median)]
        else:
            return []

    else:
        if nodes[root_index].sa is not None and nodes[root_index].sa[1] == attr:
            ans = [(root_index, nodes[root_index].median)]
        else:
            ans = []

        max_temp = []
        for key in graph[root_index]:
            temp = num_of_times_used_as_split(graph[root_index][key], attr)
            if(len(temp) > len(max_temp)):
                max_temp = temp
        return ans + max_temp


def num_of_times(root_index, attributes):
    for attrib in attributes:
        print(attrib, num_of_times_used_as_split(root_index, attrib[1]))


train_data = rd.preprocess("dataset/dtree_data/train.csv", binarize=False)
train_labels = np.array(train_data[:, 0])
train_data = np.delete(train_data, 0, 1)
test_data = rd.preprocess("dataset/dtree_data/test.csv", binarize=False)
test_labels = np.array(test_data[:, 0])
test_data = np.delete(test_data, 0, 1)
valid_data = rd.preprocess("dataset/dtree_data/valid.csv", binarize=False)
valid_labels = np.array(valid_data[:, 0])
valid_data = np.delete(valid_data, 0, 1)


build_tree(train_data, train_labels, [i for i in range(len(train_data))], list(attributes))
bfs_order = bfs(0)
bfs_fast = np.zeros(len(nodes))
for i in range(len(nodes)):
    bfs_fast[bfs_order[i]] = 1


num_of_times(0, attributes)
print("\nTraining accuracy: %0.2f Validation accuracy: %0.2f Test accuracy: %0.2f" % accuracy(bfs_fast))

part_c(step=10)
# pruning()

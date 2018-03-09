import matplotlib.pyplot as plt
import itertools
import sys
import numpy as np

def plot_crossvalidation():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x = [10e-5, 10e-3, 1, 5, 10]
    y_cv = [71.59, 71.59, 97.355, 97.455, 97.455]
    y_test = [72.11, 72.11, 97.23, 97.29, 97.29]
    cv = plt.semilogx(x, y_cv, marker="o", label="Cross Validation")
    test = plt.semilogx(x, y_test, marker="*", color="green", label="Test Data")
    plt.xticks(x, x)
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    ax.legend()
    plt.show()

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    fig = plt.figure()
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
    plt.show()


def get_y(file_xy, file_predicted_labels):
    y_gold = []
    y_pred = []

    with open(file_xy, "r") as gold, open(file_predicted_labels, "r") as predictions:
        for xy, p_y in zip(gold, predictions):
            xy = list(map(int, xy.strip().split(',')))
            y = xy[-1]
            p_y = int(p_y.strip())

            y_gold.append(y)
            y_pred.append(p_y)
            cf_mat[y][p_y] += 1

    return y_gold, y_pred


plot_crossvalidation()
plt.show()

cf_mat = np.zeros([10, 10])
y_gold, y_pred = get_y(sys.argv[1].strip(), sys.argv[2].strip())
plot_confusion_matrix(cf_mat, [i for i in range(10)])



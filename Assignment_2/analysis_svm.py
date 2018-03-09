import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

file_xy = sys.argv[1].strip()
file_predicted_labels = sys.argv[2].strip()

with open(file_xy, "r") as gold, open(file_predicted_labels, "r") as predictions:
    correct = 0
    total = 0
    for xy, p_y in zip(gold, predictions):
        xy = list(map(int, xy.strip().split(',')))
        x = xy[:-1]
        y = xy[-1]
        p_y = int(p_y.strip())

        total += 1
        if (y == p_y):
            correct += 1
        else:
            print(y, p_y)
            plt.imshow(np.array(x).reshape(28, 28), cmap=cm.gray, vmin=0, vmax=255)
            plt.show()

    print(correct / total)

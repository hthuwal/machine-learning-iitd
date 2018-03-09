import matplotlib.pyplot as plt

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

plot_crossvalidation()
plt.show()
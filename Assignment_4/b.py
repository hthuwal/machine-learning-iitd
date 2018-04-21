from a import load_data
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
from sklearn.svm import SVC
import pickle


def save_to_file(pred, file):
    with open(file, "w") as f:
        f.write("ID,CATEGORY\n")
        for i in range(len(pred)):
            f.write("%d,%s\n" % (i, pred[i]))


def do_gridsearch(model, params):
    print("Doing Cross Validation\n")
    print(model)
    print(params)
    grid = GridSearchCV(model, params, verbose=2, n_jobs=-1, cv=10)
    grid.fit(smalld_train_data, train_labels)
    return grid


def train_and_predict(model, model_file, pred_file):
    print(model)
    print("\nTraining model\n")
    model.fit(smalld_train_data, train_labels)
    print("\nSaving Model\n")
    pickle.dump(model, open(model_file, "wb"))
    print("\nMaking Predictions\n")
    pred = model.predict(smalld_test_data)
    print("\nSaving Predictions to file\n")
    save_to_file(pred, pred_file)


train_data, train_labels = load_data("dataset/train")
test_data, test_labels = load_data("dataset/test")

print("Scaling Data")
train_data = scale(train_data)
test_data = scale(test_data)

print("PCA")
pca = PCA(n_components=50)  # TODO play with other parameters
pca.fit(train_data)

smalld_train_data = pca.transform(train_data)
smalld_test_data = pca.transform(test_data)

del train_data
del test_data


# Linear SVC

# Grid Search
# params = {
#     'C': [1e-5, 1e-3, 1e-2, 1e-4, 1e-1, 1, 5, 10.0]
# }

# model = SVC(kernel='linear', decision_function_shape='ovo')
# grid = do_gridsearch(model, params)

# best value of c after gridsearch is c = 0.001, cross validation accuracy of 69
# model = SVC(C=0.001, kernel='linear', decision_function_shape='ovo', verbose=1)
# train_and_predict(model, "models/b/best_svm_linear.model", "outputs/b/out_svm_linear.txt")

# RBF SVC
model = SVC(gamma=0.05, C=0.001, verbose=1)
train_and_predict(model, "models/b/rbf_g_0.05_c_0.01.model", "outputs/b/out_rbf_g_0.05_c_0.01.txt")

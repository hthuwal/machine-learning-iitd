from a import load_data
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
from sklearn.svm import SVC

train_data, train_labels = load_data("dataset/train")
test_data, test_labels = load_data("dataset/test")

print("Scalind Data")
train_data = scale(train_data)
test_data = scale(test_data)

print("PCA")
pca = PCA(n_components=50)  # TODO play with other parameters
pca.fit(train_data)

smalld_train_data = pca.transform(train_data)
smalld_test_data = pca.transform(test_data)

del train_data
del test_data

model = SVC(kernel='linear', decision_function_shape='ovo', verbose=1)
# model.fit(smalld_train_data, train_labels)

# For last part try rbf
params = {
    'C': [1e-5, 1e-3, 1e-2, 1e-4, 1e-1, 1, 5, 10.0]
}

print("Doing Cross Validation")
grid = GridSearchCV(model, params, verbose=2, n_jobs=8, cv=10)
grid.fit(smalld_train_data, train_labels)

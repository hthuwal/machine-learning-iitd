from a import load_data
from sklearn.decomposition import PCA

train_data, train_labels = load_data("dataset/train")
test_data, test_labels = load_data("dataset/test")

pca = PCA(n_components=50)  # TODO play with other parameters
pca.fit(train_data)

smalld_train_data = pca.transform(train_data)
smalld_test_data = pca.transform(test_data)

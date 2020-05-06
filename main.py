import numpy as np
from utils import Get_training_data, Get_testing_data
from sklearn.decomposition import KernelPCA
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt 

X = Get_training_data()
transformer = KernelPCA(n_components= 8, kernel='rbf')

X_pca = []
for x in X:
	print(np.array(x).shape)
	transformed = transformer.fit_transform(x)
	for i in transformed:
		X_pca.append(i)
clf = OneClassSVM(gamma='auto')
X_pca = np.array(X_pca)
print(np.array(X_pca).shape)
plt.scatter(X_pca[:, 0], X_pca[:, 1], label="train data")
plt.show()
clf.fit(X_pca)
print(clf.predict(X_pca))
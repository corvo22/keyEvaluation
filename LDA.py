import numpy as np

class LDA:

	def __init__(self, n_componenets):
		# number of dimmensions to keep
		# disrciminants = store for eigenvectors 

		self.n_componenets = n_componenets
		self.discriminants = None

	def fit(self, X, y):
		num_features = X.shape[1]
		class_labels = np.unique(y)

		# calc S_W and S_B

		mean_over_all = np.mean(X,axis=0)

		SW = np.zeros((num_features,num_features))
		SB = np.zeros((num_features,num_features))

		for c in class_labels:
			X_c = X[y==c]
			mean_c = np.mean(X_c,axis=0)
			SW += (X_c - mean_c).T.dot(X_c - mean_c)

			n_C = X_c.shape[0]
			mean_diff = (mean_c - mean_over_all).reshape(num_features,1)
			SB += n_C * (mean_diff).dot(mean_diff.T)

		A = np.linalg.inv(SW).dot(SB)
		eigenvalues, eigenvectors = np.linalg.eig(A)
		eigenvectors = eigenvectors.T
		idxs = np.argsort(abs(eigenvalues))[::-1]
		eigenvalues = eigenvalues[idxs]
		eigenvectors = eigenvectors[idxs]

		self.discriminants = eigenvectors[0:self.n_componenets]

	def transform(self, X):
		# project down dimmensions

		return np.dot(X, self.discriminants.T)

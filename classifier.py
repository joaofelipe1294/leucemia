from sklearn.neighbors import KNeighborsClassifier


class Classifier(object):


	def __init__(self , X_train , y_train , X_valid):
		self.X_train = X_train
		self.y_train = y_train
		self.X_valid = X_valid
		

	def knn(self , k = 3):
		self.classifier = KNeighborsClassifier(n_neighbors = k)
		self.classifier.fit(self.X_train, self.y_train)
		classes = self.classifier.predict(self.X_valid)
		return classes		
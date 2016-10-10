from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

class Classifier(object):


	def __init__(self , X_train , y_train , X_valid):
		self.X_train = X_train
		self.y_train = y_train
		self.X_valid = X_valid
		

	def train_and_predict(self):
		self.classifier.fit(self.X_train, self.y_train)  #traina o classificador
		classes = self.classifier.predict(self.X_valid)  #preve as classe para cada uma das entradas
		return classes                                   #retorna as classes que foram 


	def knn(self , k = 3):
		self.classifier = KNeighborsClassifier(n_neighbors = k)
		return self.train_and_predict()
		

	def svm(self):
		self.classifier = SVC(kernel="linear" , C = 0.025)
		return self.train_and_predict()
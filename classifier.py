from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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


	def lda(self):
		self.classifier = LinearDiscriminantAnalysis()
		return self.train_and_predict()


	def adaboost(self):
		self.classifier =  AdaBoostClassifier()
		return self.train_and_predict()


	def randon_forest(self):
		self.classifier = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
		return self.train_and_predict()


	def decision_tree(self):
		self.classifier = tree.DecisionTreeClassifier()
		return self.train_and_predict()

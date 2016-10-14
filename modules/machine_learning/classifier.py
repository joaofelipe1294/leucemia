from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class Classifier(object):


	def __init__(self , X_train , y_train , X_validation , probability = False):
		self.X_train = X_train
		self.y_train = y_train
		self.X_validation = X_validation
		self.probability = probability


	def train_and_predict_label(self):
		self.classifier.fit(self.X_train, self.y_train)  #traina o classificador
		if self.probability:
			return self.classifier.predict_proba(self.X_validation)  #preve as classe para cada uma das entradas
		else:
			return self.classifier.predict(self.X_validation)


	def knn(self , k = 3):
		self.classifier = KNeighborsClassifier(n_neighbors = k)
		return self.train_and_predict_label()
		

	def svm(self):
		if self.probability:
			self.classifier = SVC(kernel="linear" , C = 0.025 , probability = True)
		else:
			self.classifier = SVC(kernel="linear" , C = 0.025)
		return self.train_and_predict_label()


	def lda(self):
		self.classifier = LinearDiscriminantAnalysis()
		return self.train_and_predict_label()


	def adaptative_boost(self):
		self.classifier =  AdaBoostClassifier()
		return self.train_and_predict_label()


	def random_forest(self):
		self.classifier = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
		return self.train_and_predict_label()


	def decision_tree(self):
		self.classifier = tree.DecisionTreeClassifier(max_depth = 5)
		return self.train_and_predict_label()

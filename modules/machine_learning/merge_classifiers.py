from classifier import Classifier

class MergeClassifiers(object):


	def __init__(self , X_train , y_train , X_valid):
		self.X_train = X_train
		self.y_train = y_train
		self.X_valid = X_valid


	def use_classifier(self , classifier_name = ''):
		if classifier_name == 'SVM':
			return Classifier(self.X_train , self.y_train , self.X_valid).svm()
		elif classifier_name == 'LDA':
			return Classifier(self.X_train , self.y_train , self.X_valid).lda()
		elif classifier_name == 'ADABOOST':
			return Classifier(self.X_train , self.y_train , self.X_valid).adaptative_boost()
		elif classifier_name == 'RFOREST':
			return Classifier(self.X_train , self.y_train , self.X_valid).random_forest()
		elif classifier_name == 'KNN':
			return Classifier(self.X_train , self.y_train , self.X_valid).knn()
		elif classifier_name == 'TREE':
			return Classifier(self.X_train , self.y_train , self.X_valid).decision_tree()
		else:
			raise Exception('Classificador invalido : ' + classifier_name)


	def use_classifier_probabilities(self , classifier_name = ''):
		if classifier_name == 'SVM':
			return Classifier(self.X_train , self.y_train , self.X_valid , probability = True).svm()
		elif classifier_name == 'LDA':
			return Classifier(self.X_train , self.y_train , self.X_valid , probability = True).lda()
		elif classifier_name == 'ADABOOST':
			return Classifier(self.X_train , self.y_train , self.X_valid , probability = True).adaptative_boost()
		elif classifier_name == 'RFOREST':
			return Classifier(self.X_train , self.y_train , self.X_valid , probability = True).random_forest()
		elif classifier_name == 'KNN':
			return Classifier(self.X_train , self.y_train , self.X_valid , probability = True).knn()
		elif classifier_name == 'TREE':
			return Classifier(self.X_train , self.y_train , self.X_valid , probability = True).decision_tree()
		else:
			raise Exception('Classificador invalido : ' + classifier_name)


	def vote(self , *classifiers):
		#metodo que aplica fusao de classificadores por meio do voto majoritario
		predictions = []
		for classifier_name in classifiers:
			predictions.append(self.use_classifier(classifier_name))
		classes = []
		for index in xrange(0 , len(predictions[0])):
			positive_votes = 0
			negative_votes = 0
			for x in xrange(0 , len(predictions)):
				if predictions[x][index] == 1:
					positive_votes += 1
				elif predictions[x][index] == 0:
					negative_votes += 1
			if positive_votes > negative_votes:
				classes.append(1)
			elif negative_votes > positive_votes:
				classes.append(0)
		return classes


	def sum(self , *classifiers):
		predictions = []
		for classifier_name in classifiers:
			predictions.append(self.use_classifier_probabilities(classifier_name))
		classes = []
		for index in xrange(0 , len(predictions[0])):
			positive_rate = 0
			negative_rate = 0
			for x in xrange(0 , len(predictions)):
				positive_rate += predictions[x][index][1]
				negative_rate += predictions[x][index][0]
			if positive_rate > negative_rate:
				classes.append(1)
			elif negative_rate > positive_rate:
				classes.append(0)
		return classes

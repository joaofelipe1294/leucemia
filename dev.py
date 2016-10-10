from base_loader import BaseLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from classifier import Classifier


base = BaseLoader(train_base_path = 'ALL_IDB2/img' ,  valid_base_path = 'Teste_ALL_IDB2/ALL')
base.load()
X = base.train_vectors
y = base.train_labels


#clf = KNeighborsClassifier(n_neighbors = 3)

#clf = SVC(kernel="linear" , C = 0.025)

#clf = LinearDiscriminantAnalysis()

#clf = tree.DecisionTreeClassifier()

#clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)

#clf = AdaBoostClassifier()


#clf.fit(X, y)
#classes = clf.predict(base.valid_vectors)


classes = Classifier(X , y , base.valid_vectors).knn()


corrects = 0
errors = 0
fn = 0
fp = 0
for index in xrange(0 , len(classes)):
	if classes[index] == base.valid_labels[index]:
		corrects += 1
	else:
		if classes[index] == 1:
			fn += 1
		elif classes[index] == 0:
			fp += 1
		errors += 1
percentage = (corrects * 100) / len(base.valid_vectors)
print('+%-15s+%-15s+%-15s+%-15s+' % ('-' * 15 , '-' * 15 , '-' * 15 , '-' * 15))
print('|%-15s|%-15s|%-15s|%-15s|' % (' ' * 15 , 'PRECISION_%' , 'FALSE_POSITIVE' , 'FALSE_NEGATIVE'))
print('+%-15s+%-15s+%-15s+%-15s+' % ('-' * 15 , '-' * 15 , '-' * 15 , '-' * 15))
print('|%-15s|%-15s|%-15s|%-15s|' % ('RESULTS' ,(' ' * 4 ) + str(percentage) , (' ' * 4 ) + str(fp) , (' ' * 4 ) + str(fn)))
print('+%-15s+%-15s+%-15s+%-15s+' % ('-' * 15 , '-' * 15 , '-' * 15 , '-' * 15))
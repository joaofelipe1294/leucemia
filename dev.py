from base_loader import BaseLoader
from sklearn.neighbors import KNeighborsClassifier


#base = BaseLoader(valid_base_path = 'validacao')
#base = BaseLoader(train_base_path = "ALL_IDB2/img" , valid_base_path = 'validacao')
#base = BaseLoader('teste')
#base = BaseLoader('ALL_IDB2/img')
#base = BaseLoader('Teste_ALL_IDB2/V0')
#base = BaseLoader('Teste_ALL_IDB2/V0')
base = BaseLoader(train_base_path = 'ALL_IDB2/img' ,  valid_base_path = 'Teste_ALL_IDB2/ALL')
#base = BaseLoader(valid_base_path = 'Teste_ALL_IDB2/ALL')
#base = BaseLoader('teste_validacao')
base.load()
X = base.train_vectors
y = base.train_labels
#print(X)
#print(y)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)
classes = neigh.predict(base.valid_vectors)
print(classes)


corrects = 0
errors = 0
for index in xrange(0 , len(classes) - 1):
	if classes[index] == base.valid_labels[index]:
		corrects += 1
	else:
		errors += 1
print("Corrects : " + str(corrects))
print("ERRORS : " + str(errors))
percentage = (corrects * 100) / len(base.valid_vectors)
print("PERCENTAGE : " + str(percentage))
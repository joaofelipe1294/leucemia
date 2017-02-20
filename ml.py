from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def read_values(file_path):
	file = open(file_path)
	X = []
	y = []
	for line in file:
		values = line.split(',')
		atributes = values[:len(values) - 1]
		atributes = [float(f) for f in atributes]
		label = int(float(values[len(values) - 1]))
		X.append(atributes)
		y.append(label)
	return X , y


train_file_path = 'treino.csv'
valid_file_path = 'validacao.csv'
X_train , y_train = read_values(train_file_path)
X_valid , y_valid = read_values(valid_file_path)


classifier = SVC(kernel="linear" , C = 0.025 , probability = True)
classifier.fit(X_train , y_train)
labels = classifier.predict(X_valid)
index = 0
hits = 0
errors = 0
while index < len(labels):
	if labels[index] == y_valid[index]:
		hits += 1
	else:
		errors += 1
	index += 1

print(labels)
print('Hits : ' + str(hits))
print('Errors : ' + str(errors))
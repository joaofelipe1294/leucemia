from base_loader import BaseLoader
from classifier import Classifier
from merge_classifiers import MergeClassifiers


base = BaseLoader(train_base_path = 'ALL_IDB2/img' ,  valid_base_path = 'Teste_ALL_IDB2/ALL')
base.load()
X = base.train_vectors
y = base.train_labels


classes = MergeClassifiers(X , y , base.valid_vectors).vote('SVM' , 'LDA' , 'TREE')
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
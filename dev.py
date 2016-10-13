from modules.base.base_loader import BaseLoader
from modules.machine_learning.classifier import Classifier
from modules.machine_learning.merge_classifiers import MergeClassifiers
from modules.base.base_processor import BaseProcessor
from modules.utils.file_handler import FileHandler


#base = BaseLoader(train_base_path = 'bases/ALL_IDB2/img' ,  validation_base_path = 'bases/Teste_ALL_IDB2/ALL')
#base.load()
#BaseProcessor().load(images = base.train_images , train = True)
#print()
#BaseProcessor().load(images = base.validation_images , validation = True)


#X = base.train_vectors
#y = base.train_labels
X , y = FileHandler().load_vectors_and_labels(train = True)
X_ , y_ = FileHandler().load_vectors_and_labels(validation = True)

#classes = MergeClassifiers(X , y , X_).vote('SVM' , 'LDA' , 'TREE')
classes = Classifier(X , y , X_).lda()

corrects = 0
errors = 0
fn = 0
fp = 0
for index in xrange(0 , len(classes)):
	if classes[index] == y_[index]:
		corrects += 1
	else:
		if classes[index] == 1:
			fn += 1
		elif classes[index] == 0:
			fp += 1
		errors += 1
percentage = (corrects * 100) / len(X_)
print('+%-15s+%-15s+%-15s+%-15s+' % ('-' * 15 , '-' * 15 , '-' * 15 , '-' * 15))
print('|%-15s|%-15s|%-15s|%-15s|' % (' ' * 15 , 'PRECISION_%' , 'FALSE_POSITIVE' , 'FALSE_NEGATIVE'))
print('+%-15s+%-15s+%-15s+%-15s+' % ('-' * 15 , '-' * 15 , '-' * 15 , '-' * 15))
print('|%-15s|%-15s|%-15s|%-15s|' % ('RESULTS' ,(' ' * 4 ) + str(percentage) , (' ' * 4 ) + str(fp) , (' ' * 4 ) + str(fn)))
print('+%-15s+%-15s+%-15s+%-15s+' % ('-' * 15 , '-' * 15 , '-' * 15 , '-' * 15))
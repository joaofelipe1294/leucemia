from modules.base.base_loader import BaseLoader
from modules.machine_learning.classifier import Classifier
from modules.machine_learning.merge_classifiers import MergeClassifiers
from modules.base.base_processor import BaseProcessor
from modules.utils.file_handler import FileHandler
from modules.base.kfold import Kfold

kfolds =  Kfold(k = 2 , base_path = 'bases/ALL')
base = BaseLoader(train_base_path =  kfolds.train_path,  validation_base_path = kfolds.validation_path)
#base = BaseLoader(train_base_path = 'bases/ALL_IDB2/img' ,  validation_base_path = 'bases/Teste_ALL_IDB2/ALL')
base.load()
BaseProcessor().load(images = base.train_images , train = True)
print()
BaseProcessor().load(images = base.validation_images , validation = True)


#X = base.train_vectors
#y = base.train_labels
X , y = FileHandler().load_vectors_and_labels(train = True)
X_ , y_ = FileHandler().load_vectors_and_labels(validation = True)

#classes = MergeClassifiers(X , y , X_).sum('KNN')
classes = Classifier(X , y , X_).svm()

corrects = 0
errors = 0
false_negative = 0
false_positive = 0
for index in xrange(0 , len(classes)):
	if classes[index] == y_[index]:
		corrects += 1
	else:
		if classes[index] == 1:
			false_negative += 1
		elif classes[index] == 0:
			false_positive += 1
		errors += 1
correct_percentage = (corrects * 100) / len(X_)
false_positive_percentage = (false_positive * 100) / len(X_)
false_negative_percentage = (false_negative * 100) / len(X_)
print('+%-15s+%-15s+%-15s+%-15s+' % ('-' * 15 , '-' * 15 , '-' * 15 , '-' * 15))
print('|%-15s|%-15s|%-15s|%-15s|' % (' ' * 15 , 'PRECISION_%' , 'FALSE_POSITIVE' , 'FALSE_NEGATIVE'))
print('+%-15s+%-15s+%-15s+%-15s+' % ('-' * 15 , '-' * 15 , '-' * 15 , '-' * 15))
print('|%-15s|%-15s|%-15s|%-15s|' % ('RESULTS' ,(' ' * 4 ) + str(correct_percentage) , (' ' * 4 ) + str(false_positive_percentage) , (' ' * 4 ) + str(false_negative_percentage)))
print('+%-15s+%-15s+%-15s+%-15s+' % ('-' * 15 , '-' * 15 , '-' * 15 , '-' * 15))

kfolds.remove_temp_dirs()
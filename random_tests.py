import cv2
import numpy as np
from modules.base.base_loader import BaseLoader
from modules.image_processing.segmentation import SmartSegmentation
from modules.utils.progress_bar import ProgressBar
from modules.image_processing.image_chanels import ImageChanels
from modules.image_processing.filters import *
from modules.features.features import Features
from modules.image_processing.segmentation import SegmentNucleus
from modules.features.features import MinMax
import os

from sklearn.svm import SVC

def extrcat_features(image_path):
	rgb_image = cv2.imread(image_path)
	segmented_image = SmartSegmentation(rgb_image).process()
	features = Features(segmented_image)
	mean = features.mean()
	median = features.median()
	standard_deviation = features.standard_deviation()
	cell_proportion = features.nucleus_proportion(rgb_image)
	return mean , median , standard_deviation , cell_proportion


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







base_path = 'bases/k_fold'
folds = [f for f in sorted(os.listdir(base_path))]

for fold in folds:
	base = BaseLoader(train_base_path = base_path + '/' + fold ,  validation_base_path = 'bases/teste/validacao')
	base.load()
	file = open( fold + '.csv', 'w')
	iteration = 1
	data = []
	for image in base.train_images:
		mean , median , standard_deviation , cell_proportion = extrcat_features(image.path)
		#data.append([mean , median , standard_deviation , cell_proportion , image.label])
		data.append([mean , cell_proportion , image.label])
		ProgressBar().printProgress(iteration , len(base.train_images) , prefix = fold + " : ")
		iteration += 1
	for value in data:
		#file.write(str(value[0]) + ',' + str(value[1]) + ',' + str(value[2]) + ',' + str(value[3]) + ',' + str(value[4]) + '\n')
		file.write(str(value[0]) + ',' + str(value[1]) + ',' + str(value[2]) + '\n')
	file.close()

#recupera os dados de cada fold , concatenaos e os normaliza
data = []
for file_name in folds:
	file = open(file_name + '.csv' , 'r')
	for line in file:
		values = line.split(',')
		data.append(values)	
normalized_data = MinMax(data).normalize(maximum = 10) 
lines_per_fold = len(data) / len(folds)

#divide os valores de cada fold baseado no numero de imagens que cada fold contem
fold_index = 0
start_index = 0
stop_index = 0
while fold_index < len(folds):
	stop_index += lines_per_fold
	file = open( folds[fold_index] + '.csv', 'w')
	while start_index < stop_index:
		value = normalized_data[start_index]
		#file.write(str(value[0]) + ',' + str(value[1]) + ',' + str(value[2]) + ',' + str(value[3]) + ',' + str(value[4]) + '\n')
		file.write(str(value[0]) + ',' + str(value[1]) + ',' + str(value[2]) + '\n')
		start_index += 1
	file.close()
	fold_index += 1

#print(len(data))



####################################################################################################
#LOOP DE TREINO E VALIDACAO
####################################################################################################




for fold_index in xrange(0 , len(folds)):
	temp_folds = list(folds)
	validation_fold = temp_folds.pop(fold_index)
	X_validation , y_validation = read_values(validation_fold + '.csv')
	X_train = []
	y_train = []
	for fold in temp_folds:
		X_t , y_t = read_values(fold + '.csv')
		X_train += X_t
		y_train += y_t
	classifier = SVC(kernel="linear" , C = 0.025 , probability = True)
	classifier.fit(X_train , y_train)
	labels = classifier.predict(X_validation)
	index = 0
	hits = 0
	errors = 0
	while index < len(labels):
		if labels[index] == y_validation[index]:
			hits += 1
		else:
			errors += 1
		index += 1

	#print(labels)
	print(validation_fold)
	print('Hits : ' + str(hits))
	print('Errors : ' + str(errors))
	precision = (hits * 100) / len(labels)
	print('Precision : ' + str(precision))
	print('===================')
	#print(labels)

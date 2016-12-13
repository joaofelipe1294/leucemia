import cv2
import numpy as np
from modules.base.base_loader import BaseLoader
from modules.image_processing.segmentation import SmartSegmentation
from modules.utils.progress_bar import ProgressBar
from modules.image_processing.image_chanels import ImageChanels
from modules.image_processing.filters import *
from modules.features.features import Features
from modules.image_processing.segmentation import SegmentNucleus



def min_max(values , minimum = 0, maximum = 1):
	values = [[float(j) for j in column] for column in values] #converto os valores para float
	values = np.array(values)
	normalized_values = np.zeros((values.shape[0] , values.shape[1] - 1))
	for column in xrange(0,values.shape[1] - 1) :
		column_values = values[:,column]
		bigger = max(column_values) #pega o maior valor da coluna
		lower = min(column_values)  #pega o menor valor da coluna
		line = 0
		for value in column_values:
			normalized_value = 0
			if value != 0:
				normalized_value = minimum + ((value - lower) / (bigger - lower)) * (maximum - minimum)
				normalized_values.itemset(line , column , normalized_value)
			line += 1
	labels = values[:,values.shape[1] - 1]
	labels = [[label] for label in labels]
	normalized_values = np.append(normalized_values , np.array(labels) , axis=1)
	return normalized_values


#treinamento
base = BaseLoader(train_base_path = 'bases/teste/treino' ,  validation_base_path = 'bases/teste/validacao')
base.load()
file = open( 'treino.csv', 'w')
iteration = 1
data = []
for image in base.train_images:
	rgb_image = cv2.imread(image.path)
	segmented_image = SmartSegmentation(rgb_image).process()
	features = Features(segmented_image)
	mean = features.mean()
	median = features.median()
	standard_deviation = features.standard_deviation()
	cell_proportion = features.nucleus_proportion(rgb_image)
	#data.append([mean , median , standard_deviation , cell_proportion , image.label])
	data.append([mean , median , standard_deviation , cell_proportion , image.label])
	ProgressBar().printProgress(iteration , len(base.train_images) , prefix = "Treinamento : ")
	iteration += 1
normalized_data = min_max(data , maximum = 10)
for value in normalized_data:
	#file.write(str(value[0]) + ',' + str(value[1]) + ',' + str(value[2]) + ',' + str(value[3]) + '\n')
	file.write(str(value[0]) + ',' + str(value[1]) + ',' + str(value[2]) + ',' + str(value[3]) + ',' + str(value[4]) + '\n')
file.close()


file = open( 'validacao.csv', 'w')
iteration = 1
data = []
for image in base.validation_images:
	rgb_image = cv2.imread(image.path)
	segmented_image = SmartSegmentation(rgb_image).process()
	features = Features(segmented_image)
	mean = features.mean()
	median = features.median()
	standard_deviation = features.standard_deviation()
	cell_proportion = features.nucleus_proportion(rgb_image)
	#data.append([mean , median , standard_deviation , cell_proportion , image.label])
	data.append([mean , median , standard_deviation , cell_proportion,  image.label])
	ProgressBar().printProgress(iteration , len(base.train_images) , prefix = "Validacao : ")
	iteration += 1
normalized_data = min_max(data , maximum = 10)
for value in normalized_data:
	#file.write(str(value[0]) + ',' + str(value[1]) + ',' + str(value[2]) + ',' + str(value[3]) + '\n')
	file.write(str(value[0]) + ',' + str(value[1]) + ',' + str(value[2]) + ',' + str(value[3]) + ',' + str(value[4]) + '\n')
file.close()















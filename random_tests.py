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


def get_nucleus(rgb_image):
	nucleus_image = SegmentNucleus(rgb_image).process()
	contours_image , contours, hierarchy = cv2.findContours(nucleus_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #metodo usado para recuperar os contornos de uma imagem binaria
 	contours_area = 0
 	for contour in contours:
 		contours_area = cv2.contourArea(contour)
 	return contours , contours_area


def get_cell_contours(segmented_image):
	gray_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
	otsu = OtsuThreshold(gray_image).process()
	contours_image , contours, hierarchy = cv2.findContours(otsu.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #metodo usado para recuperar os contornos de uma imagem binaria
	contours_area = 0
	for contour in  contours:
		contours_area += cv2.contourArea(contour)
	return contours , contours_area
	



#treinamento
base = BaseLoader(train_base_path = 'bases/ALL' ,  validation_base_path = 'bases/Teste_ALL_IDB2/ALL')
base.load()
file = open( 'values_norm_2.csv', 'w')
iteration = 1
data = []
for image in base.train_images:
	rgb_image = cv2.imread(image.path)
	segmented_image = SmartSegmentation(rgb_image).process()
	#mean , median , standard_deviation = get_statistics_features(segmented_image)
	features = Features(segmented_image)
	mean = features.mean()
	median = features.median()
	standard_deviation = features.standard_deviation()
	cell_proportion = features.nucleus_proportion(rgb_image)
	
	#cell_contours , cell_area = get_cell_contours(segmented_image)
	#nucleus_contours , nucleus_area = get_nucleus(rgb_image)
	#if cell_area < nucleus_area:
	#	cell_area = nucleus_area + 1000
	data.append([mean , median , standard_deviation , cell_proportion , image.label])
	ProgressBar().printProgress(iteration , len(base.train_images) , prefix = "Progresso : ")
	iteration += 1
	#cv2.imwrite('bases/ALL_RESULTS/LDA/Im' + image.path[12:] + 'f',segmented_image)
#print(data)

normalized_data = min_max(data , maximum = 100)
for value in normalized_data:
	file.write(str(value[0]) + ',' + str(value[1]) + ',' + str(value[2]) + ',' + str(value[3]) + ',' + str(value[4]) + '\n')
file.close()
















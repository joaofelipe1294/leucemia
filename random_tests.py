import cv2
import numpy as np
from modules.base.base_loader import BaseLoader
from modules.image_processing.segmentation import SmartSegmentation
from modules.utils.progress_bar import ProgressBar


def get_statistics_features(rgb_image):
	gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
	reshaped_image = gray_image.reshape(-1)
	clean_image = reshaped_image[reshaped_image != 0]
	mean = 0
	median = 0
	standard_deviation = 0
	if len(clean_image) > 0:
		mean = int(np.mean(clean_image))
		median = int(np.median(clean_image))
		standard_deviation = np.std(clean_image)
	else:
		mean = 0
	return mean , median , standard_deviation


def min_max(values , minimum = 0, maximum = 1):
	values = [[float(j) for j in column] for column in values] #converto os valores para float
	normalized_values = []
	for column in values:
		bigger = max(column) #pega o maior valor da coluna
		lower = min(column)  #pega o menor valor da coluna
		normalized_column = []
		for value in column:
			normalized_value = minimum + ((value - lower) / (bigger - lower)) * (maximum - minimum)
			normalized_column.append(normalized_value)
		normalized_values.append(normalized_column)
	return normalized_values

'''
base = BaseLoader(train_base_path = 'bases/ALL' ,  validation_base_path = 'bases/Teste_ALL_IDB2/ALL')
base.load()
file = open( 'values.csv', 'w')
iteration = 1
for image in base.train_images:
	segmented_image = SmartSegmentation(image.path).process()
	mean , median , standard_deviation = get_statistics_features(segmented_image)
	file.write(str(mean) + ',' + str(median) + ',' + str(standard_deviation) + ',' + str(image.label) + '\n')
	ProgressBar().printProgress(iteration , len(base.train_images) , prefix = "Progresso : ")
	iteration += 1
file.close()
'''











values = ([900 , 800 , 600 , 550 , 755] , [7,5,2,5,7] , [12, 10 , 12 , 10 , 10] , [-3 , 0 , 0 , -1 , 2])
normalized_values = min_max(values , maximum = 10)
print(normalized_values)
































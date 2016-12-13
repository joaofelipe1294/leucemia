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





base_path = 'bases/k_fold'
folds = [f for f in sorted(os.listdir(base_path))]

for fold in folds:
	base = BaseLoader(train_base_path = base_path + '/' + fold ,  validation_base_path = 'bases/teste/validacao')
	base.load()
	file = open( fold + '.csv', 'w')
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
		data.append([mean , median , standard_deviation , cell_proportion , image.label])
		ProgressBar().printProgress(iteration , len(base.train_images) , prefix = fold + " : ")
		iteration += 1
	for value in data:
		file.write(str(value[0]) + ',' + str(value[1]) + ',' + str(value[2]) + ',' + str(value[3]) + ',' + str(value[4]) + '\n')
	file.close()


data = []
for file_name in folds:
	file = open(file_name + '.csv' , 'r')
	for line in file:
		values = line.split(',')
		data.append(values)	
normalized_data = MinMax(data).normalize(maximum = 10) 
lines_per_fold = len(data) / len(folds)

fold_index = 0
start_index = 0
stop_index = 0
print(lines_per_fold)
while fold_index < len(folds):
	stop_index += lines_per_fold
	file = open( folds[fold_index] + '.csv', 'w')
	while start_index < stop_index:
		value = normalized_data[start_index]
		file.write(str(value[0]) + ',' + str(value[1]) + ',' + str(value[2]) + ',' + str(value[3]) + ',' + str(value[4]) + '\n')
		start_index += 1
	file.close()
	fold_index += 1

#print(len(data))


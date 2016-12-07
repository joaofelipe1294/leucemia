import cv2
import numpy as np
from modules.base.base_loader import BaseLoader
from modules.image_processing.segmentation import SmartSegmentation
from modules.utils.progress_bar import ProgressBar


base = BaseLoader(train_base_path = 'bases/ALL' ,  validation_base_path = 'bases/Teste_ALL_IDB2/ALL')
base.load()
file = open( 'values.csv', 'w')
iteration = 1
for image in base.train_images:
	segmented_image = SmartSegmentation(image.path).process()
	gray_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
	reshaped_image = gray_image.reshape(-1)
	clean_image = reshaped_image[reshaped_image != 0]
	mean = 0
	if len(clean_image) > 0:
		mean = int(np.mean(clean_image))
		median = int(np.median(clean_image))
		standard_deviation = np.std(clean_image)
	else:
		mean = 0
	#print('mean' + str(mean) + ' | len : ' + str(len(clean_image)))
	#if mean :
	#	mean = int(mean)
	#else:
	#	mean = 0
	file.write(str(mean) + ',' + str(median) + ',' + str(standard_deviation) + ',' + str(image.label) + '\n')
	ProgressBar().printProgress(iteration , len(base.train_images) , prefix = "Progresso : ")
	iteration += 1
	#print(int(np.mean(clean_image)))
	#cv2.imshow('imagem_segmentada' , segmented_image)
	#cv2.imshow('gray_image' , gray_image)
	#cv2.waitKey(0)

file.close()
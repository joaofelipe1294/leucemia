import cv2
import numpy as np
import matplotlib.pyplot as plt
from base_loader import BaseLoader
from histogram import Histogram
from image_chanels import ImageChanels
from segmentation import Segmentation
from feature_extractor import FeatureExtractor
import math


base = BaseLoader('teste')
#base = BaseLoader('ALL_IDB2/img')
#base = BaseLoader('Teste_ALL_IDB2/V0')
#base = BaseLoader('Teste_ALL_IDB2/V0')
#base = BaseLoader('Teste_ALL_IDB2/ALL')
#base = BaseLoader('teste_validacao')
#values_1 = []
#values_0 = []
#labels = []
base.load()
#base.train()
#for image in base.images:
	#print("==============================================")
	#print(image.path)
	#segmented_image = Segmentation(image.path).process()
	#FeatureExtractor(segmented_image).get_features()
	#value = area_refs_min_circle(segmented_image)
	#cv2.imshow('image' , segmented_image)
	#cv2.waitKey(0)
	#value = get_cell_area(segmented_image)
	#value = get_cell_variance(segmented_image)
	#value = get_cell_perimeter(segmented_image)	
	#if image.label == '1':
	#	values_1.append(value)
	#else:
	#	values_0.append(value)
#plt.plot(values_0 , 'go' , values_1 , 'rx')
#plt.show()



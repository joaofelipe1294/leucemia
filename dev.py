import cv2
import numpy as np
import matplotlib.pyplot as plt
from base_loader import BaseLoader
from histogram import Histogram
from image_chanels import ImageChanels
from segmentation import Segmentation
from feature_extractor import FeatureExtractor
import math


"""def get_cell_area(image):
	area = 0
	height , width = image.shape[:2]
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	area = np.count_nonzero(gray_image)
	normalized_area = (area * 100) / (height * width)
	return normalized_area
	

def get_cell_variance(image):
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	variance = np.var(gray_image)
	return variance


def get_cell_perimeter(image):
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	binary_image = cv2.threshold(gray_image,1,255,cv2.THRESH_BINARY)[1]
	contours_image, contours, hierarchy = cv2.findContours(binary_image.copy() , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(contours_image, contours, -1, 255, 1)
	cell_perimeter = 0
	for contour in contours:
		cell_perimeter += int(cv2.arcLength(contour , True))
	return cell_perimeter 


def area_refs_min_circle(image):
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	binary_image = cv2.threshold(gray_image,1,255,cv2.THRESH_BINARY)[1]
	contours_image, contours, hierarchy = cv2.findContours(binary_image.copy() , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(contours_image, contours, -1, 255, 1)
	cell_area = 0
	circunference_area = 0
	for contour in contours:
		cell_area += int(cv2.contourArea(contour))
		(x,y),radius = cv2.minEnclosingCircle(contour)
		radius = int(radius)
		circunference_area += int(math.pi * (radius * radius))
	relative_area = (cell_area * 100) / circunference_area
	#print(cell_area)
	#print(circunference_area)
	#print(relative_area)
	return relative_area
	#cv2.imshow('contours' , contours_image)
	#cv2.waitKey(0)
"""



base = BaseLoader()
#base.load('teste')
base.load('ALL_IDB2/img')
#base.load('Teste_ALL_IDB2/V0')
#base.load('Teste_ALL_IDB2/V1')
#base.load('Teste_ALL_IDB2/ALL')
#base.load('teste_validacao')
values_1 = []
values_0 = []
#labels = []


for image in base.images:
	print("==============================================")
	print(image.path)
	segmented_image = Segmentation(image.path).process()
	FeatureExtractor(segmented_image).get_features()
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



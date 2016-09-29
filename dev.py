from base_loader import BaseLoader
import cv2
import numpy as np
import math
from histogram import Histogram
from image_chanels import ImageChanels
from segmentation import Segmentation


def get_number_of_objects(image):
	img , contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	return len(contours)


def verifies_intersection(cell_radius , cell_center , object_radius , object_center , image_height , image_width):
	plan_1 = np.zeros((image_height , image_width) , np.uint8)
	cv2.circle(plan_1 , cell_center , cell_radius , 100 , 1)
	plan_2 = np.zeros((image_height , image_width) , np.uint8)
	cv2.circle(plan_2 , object_center , object_radius , 100 , 1)
	plan = cv2.add(plan_1 , plan_2)
	counter = 0
	for x in xrange(0,image_height):
		for y in xrange(0,image_width):
			if plan.item(x , y) > 100:
				counter += 1	
	if counter == 2:
		return True
	else:
		return False
	

def flood_object(image , contour_image , object_center, object_radius , value=255):
	flag = False
	x_white = 0
	y_white = 0
	for index in xrange(0,object_radius):	
		x_coordenate = tuple([object_center[0] + index , object_center[0]])
		y_coordenate = tuple([object_center[0] , object_center[1] + index])
		flooded_image_X = flood(contour_image , value , x_coordenate)
		flooded_image_Y = flood(contour_image , value , y_coordenate)
		for x in xrange(0,image.shape[0]):
			for y in xrange(0,image.shape[1]):
				if flooded_image_X.item(x,y) == 255:
					x_white += 1
				if flooded_image_Y.item(x,y) == 255:
					y_white += 1
		if x_white < ((image.shape[0] * image.shape[1] * 80) / 100):
			sum_image = cv2.add(flooded_image_X , image)
		elif y_white < ((image.shape[0] * image.shape[1] * 80) / 100):
			sum_image = cv2.add(flooded_image_Y , image)
		open_image = cv2.morphologyEx(sum_image, cv2.MORPH_OPEN, np.ones((5,5) , np.uint8))	
		#show = np.concatenate((image , contour_image , sum_image , open_image , flooded_image_X , flooded_image_Y) , axis=1)
		#cv2.imshow('resault' , show)
		#cv2.waitKey(0)
		return open_image

def define_mask(image , threshold_image , contours , cell_center , cell_radius):
	flooded_image = flood(image.copy() , 255 , single_seed = cell_center)
	open_image = cv2.morphologyEx(flooded_image.copy(), cv2.MORPH_OPEN, np.ones((5,5) , np.uint8))
	cv2.circle(grayscale_image , cell_center , cell_radius , 255 , 1)
	
	for contour in contours:
		(x,y),object_radius = cv2.minEnclosingCircle(contour)
		object_radius = int(object_radius)
		object_center = (int(x),int(y))
		distance_to_cell_center = math.sqrt(math.pow((x - cell_center[1]) , 2) + math.pow(y - cell_center[1] , 2))
		cv2.circle(grayscale_image , object_center , object_radius , 155 , 1)
		if (object_center[0] + object_radius > cell_center[0] + cell_radius and object_center[0] - object_radius < cell_center[0] - cell_radius) and (object_center[1] + object_radius > cell_center[1] + cell_radius and object_center[1] - object_radius < cell_center[1] + cell_radius):
			open_image = cv2.add(threshold_image , open_image)
		intercepts = verifies_intersection(cell_radius , cell_center , object_radius , object_center , image.shape[0] , image.shape[1])
		if intercepts:
			open_image = flood_object(open_image.copy() , image , object_center , object_radius)
	return open_image
	

	




base = BaseLoader()
base.load('teste')
#base.load('ALL_IDB2/img')

for image in base.images:
	print("==============================================")
	print(image.path)
	"""rgb_image = cv2.imread(image.path)
	grayscale_image = cv2.imread(image.path ,cv2.IMREAD_GRAYSCALE)
	saturation = ImageChanels(rgb_image).hsi('S') 
   	otsu_image = otsu_threshold(saturation)
	flooded_image = flood(otsu_image)
	opening = cv2.morphologyEx(flooded_image, cv2.MORPH_OPEN, np.ones((5,5) , np.uint8))
	contours_image , contours, hierarchy = cv2.findContours(opening.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(contours_image, contours, -1,255, 1)
	image_center_point = tuple([int(opening.shape[0] / 2) , int(opening.shape[1]) / 2])
	if get_number_of_objects(opening) > 1:
		cell_center , cell_radius , contours = find_interest_cell(opening)
		if cell_radius < 10:
			mask = opening.copy()
		else:
			mask = define_mask(contours_image , opening , contours , cell_center , cell_radius)
	else:
		mask = opening.copy()
	"""
	print("==============================================")
	#cv2.imshow('opening' , opening)
	#cv2.imshow('mask' , mask)
	#cv2.imshow('contours_image' , contours_image)
	#cv2.waitKey(0)
	#print(mask.shape)
	#show = np.concatenate((grayscale_image , flooded_image , opening , contours_image , mask) , axis=1)
	#cv2.imshow('resault' , show)
	#cv2.waitKey(150)
	saturation = Segmentation(image.path).process()
	cv2.imshow('resault' , saturation)
	cv2.waitKey(0)
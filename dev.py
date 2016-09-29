from base_loader import BaseLoader
import cv2
import numpy as np
#from matplotlib import pyplot as plt
import math
from histogram import Histogram


def flood(image , value=0 , single_seed = None):
	floodfill_image = image.copy()
	h, w = floodfill_image.shape[:2]                                         
	mask = np.zeros((h + 2 , w + 2) , np.uint8)                              
	if single_seed == None:
		seeds = []
		for x in xrange(0 , w , 5):
			seeds.append(tuple([0 , x]))
			seeds.append(tuple([h - 5 , x]))
			seeds.append(tuple([x , 0]))
			seeds.append(tuple([x , w - 5]))
		for seed in seeds:
			cv2.floodFill(floodfill_image , mask , seed , value , loDiff = 2 , upDiff = 2)
	else:
		seed = single_seed
		cv2.floodFill(floodfill_image , mask , seed , value , loDiff = 2 , upDiff = 2)
	white = 0
	for x in xrange(0,image.shape[0]):
		for y in xrange(0,image.shape[1]):
			if floodfill_image.item(x,y) == 255:
				white += 1
	if white > ((image.shape[0] * image.shape[1] * 80) / 100):
		floodfill_image = cv2.bitwise_not(floodfill_image)
	return floodfill_image
	

def rgb_chanels(rgb_image):
	height , width = rgb_image.shape[:2]
	red_image = np.zeros((height , width) , np.uint8)
	green_image = np.zeros((height , width) , np.uint8)
	blue_image = np.zeros((height , width) , np.uint8)
	color_chanels = [blue_image , green_image , red_image]
	for chanel_index in range(0,3):
		for line in xrange(0,height):
			for col in xrange(0,width):
				value = rgb_image.item(line , col , chanel_index)
				color_chanels[chanel_index].itemset((line , col) , value)
	show = np.concatenate(( red_image , green_image , blue_image) , axis=1)
	cv2.imshow('resault' , show)
	cv2.imshow('rgb' , rgb_image)
	cv2.waitKey(0)


def hsi_chanels(rgb_image, chanel_return = 'S'):
	if chanel_return == 'H':
		chanel_index = 0
	elif chanel_return == 'S':
		chanel_index = 1
	elif chanel_return == 'I':
		chanel_index = 2
	hsi_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
	height , width = rgb_image.shape[:2]
	chanel = np.zeros((height , width) , np.uint8)
	for line in xrange(0 , height):
		for col in xrange(0 , width):
			val = hsi_image.item( line, col, chanel_index)
			chanel.itemset((line , col), val)
	return chanel


def otsu_threshold(image):
	blur_image = cv2.GaussianBlur(saturation,(5,5),0)
   	otsu_image = cv2.threshold(blur_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
   	return otsu_image


def get_number_of_objects(image):
	img , contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	return len(contours)


def find_interest_cell(image):
	image_center_point = tuple([int(opening.shape[0] / 2) , int(opening.shape[1]) / 2])
	img , contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	lowest_index = 0
	lowest_distance = None
	for contour_index in range(0 , len(contours)):
		(x,y),cell_radius = cv2.minEnclosingCircle(contours[contour_index])
		distance_to_center = math.sqrt(math.pow((x - image_center_point[1]) , 2) + math.pow(y - image_center_point[1] , 2))
		if lowest_distance == None or distance_to_center < lowest_distance:
			lowest_index = contour_index
			lowest_distance = distance_to_center
	(x,y),cell_radius = cv2.minEnclosingCircle(contours[lowest_index])
	cell_radius = int(cell_radius)
	contours.pop(lowest_index)
	cell_center = (int(x),int(y))
	return cell_center , cell_radius , contours


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
	rgb_image = cv2.imread(image.path)
	grayscale_image = cv2.imread(image.path ,cv2.IMREAD_GRAYSCALE)
	saturation = hsi_chanels(rgb_image)
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
	print("==============================================")
	#cv2.imshow('opening' , opening)
	#cv2.imshow('mask' , mask)
	#cv2.imshow('contours_image' , contours_image)
	#cv2.waitKey(0)
	#show = np.concatenate((grayscale_image , flooded_image , opening , contours_image , mask) , axis=1)
	#cv2.imshow('resault' , show)
	#cv2.waitKey(0)
	
	
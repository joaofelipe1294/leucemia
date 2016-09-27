from base_loader import BaseLoader
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

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
	return floodfill_image


def display_rgb_histogram(rgb_image):
	color = ('b','g','r')
	for i,col in enumerate(color):
		histr = cv2.calcHist([rgb_image],[i],None,[256],[0,256])
		plt.plot(histr,color = col)
		plt.xlim([0,256])
	plt.show()

def display_gray_scale_histogram(grayscale_image):
	plt.hist(grayscale_image.ravel(),256,[0,256])
	plt.show()
	

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



base = BaseLoader()
base.load('teste')
#base.load('ALL_IDB2/img')

for image in base.images:
	print(image.path)
	rgb_image = cv2.imread(image.path)
	grayscale_image = cv2.imread(image.path ,cv2.IMREAD_GRAYSCALE)
	saturation = hsi_chanels(rgb_image)
	blur_image = cv2.GaussianBlur(saturation,(5,5),0)
   	otsu_image = cv2.threshold(blur_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
	flooded_image = flood(otsu_image)
	opening = cv2.morphologyEx(flooded_image, cv2.MORPH_OPEN, np.ones((5,5) , np.uint8))
	img , contours, hierarchy = cv2.findContours(opening.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(img, contours, -1,255, 1)
	image_center_point = tuple([int(opening.shape[0] / 2) , int(opening.shape[1]) / 2])
	if len(contours) > 1:
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
	else:
		(x,y),cell_radius = cv2.minEnclosingCircle(contours[0])
	cell_center = (int(x),int(y))
	copy = np.zeros((opening.shape) , np.uint8)
	mask = flood(img , 255 , single_seed = cell_center)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3) , np.uint8))

		#print(len(contours))
		#for contour in contours:
			#cv2.drawContours(edges, contours, -1, (0,255,0), 3)
			#(x,y),object_radius = cv2.minEnclosingCircle(contour)
			#object_center = (int(x),int(y))
			#object_radius = int(object_radius)
			#print("========================================")
			#print("CELL CENTER : " + str(cell_center))
			#print("CELL RADIUS : " + str(cell_radius))
			#print("OBJECT CENTER : " + str(object_center))
			#print("OBJECT RADIUS : " + str(object_radius))
			#print(object_center[0] > cell_center[0] + cell_radius) 
			#print(object_center[0] < cell_center[0] - cell_radius)
			#print(object_center[1] > cell_center[1] + cell_radius or object_center[1] < cell_center[1] - cell_radius)
			#print("========================================")
			#opening.itemset(cell_center , 0)
			#if (object_center[0] > cell_center[0] + cell_radius or object_center[0] < cell_center[0] - cell_radius) and (object_center[1] > cell_center[1] + cell_radius or object_center[1] < cell_center[1] - cell_radius):
			#	cv2.circle(opening,object_center,object_radius + 5,0,-1)
		#(x,y),radius = cv2.minEnclosingCircle(contours[lowest_index])
		#center = (int(x),int(y))
		#radius = int(radius)
		#cv2.circle(rgb_image,cell_center,cell_radius,(0,255,0),2)
		#cv2.imshow('rgb' , rgb_image)
		#cv2.waitKey(0)"""

	cv2.imshow('opening' , opening)
	cv2.imshow('mask' , mask)
	#cv2.imshow('original' , rgb_image )
	cv2.imshow('img' , img)
	cv2.waitKey(0)
	#rgb_chanels(rgb_image)
	#show_histogram(image.path)
	#cv2.imshow('image' ,cv2.imread( image.path))
	#cv2.waitKey(250)
	#original_image = cv2.imread( image.path, 0)
	#equalized_image = cv2.equalizeHist(original_image)
	#flooded_image = flood(original_image)
	#laplacian = cv2.Laplacian(flooded_image , cv2.CV_8U)
	#edges = cv2.Canny(flooded_image,10,200)
	#erosion = cv2.erode(cv2.bitwise_not(edges),np.ones((2,2) , np.uint),iterations = 1)
	#dilation = cv2.dilate(flooded_image,np.ones((3,3) , np.uint),iterations = 1)
	#closing = cv2.morphologyEx(flooded_image, cv2.MORPH_CLOSE, np.ones((3,3) , np.uint8))
	#show = np.concatenate((original_image , equalized_image & erosion ,flooded_image & erosion , dilation , closing ) , axis=1)
	#cv2.imshow('resault' , show)
	#cv2.waitKey(0)"""


""" mostra todas as imagens pre processadas ""
for cell in base_loader.cells:
	gray_scale_image = cv2.imread(cell.image_path , 0)
	floodfill_image = gray_scale_image.copy()
	h, w = floodfill_image.shape[:2]                                         
	mask = np.zeros((h + 2 , w + 2) , np.uint8)                              
	print(cell.image_id)
	seeds = []
	for x in xrange(0 , 230 , 5):
		seeds.append(tuple([0 , x]))
		seeds.append(tuple([230 , x]))
		seeds.append(tuple([x , 0]))
		seeds.append(tuple([x , 230]))
	for seed in seeds:
		cv2.floodFill(floodfill_image , mask , seed , 0 , loDiff = 3 , upDiff = 3)
	show = np.concatenate((gray_scale_image , floodfill_image) , axis=1)
	cv2.imshow('resault' , show)
	cv2.waitKey(250)
""""""""""""""""""""""""""""""""""""""""""
rgb_image = cv2.imread(image_path)
gray_scale_image = cv2.imread(image_path , 0)
floodfill_image = gray_scale_image.copy()
h, w = floodfill_image.shape[:2]                                         
mask = np.zeros((h + 2 , w + 2) , np.uint8)                              
print(mask.shape)
seeds = []
for x in xrange(0 , 259 , 5):
	seeds.append(tuple([0 , x]))
	seeds.append(tuple([255 , x]))
	seeds.append(tuple([x , 0]))
	seeds.append(tuple([x , 255]))
for seed in seeds:
	cv2.floodFill(floodfill_image , mask , seed , 255 , loDiff = 4 , upDiff = 4)

#cv2.imshow('gray' , gray_scale_image)
cv2.imshow('flooded image' , floodfill_image)
cv2.waitKey(0)"""


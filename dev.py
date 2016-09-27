from base_loader import BaseLoader
import cv2
import numpy as np
from matplotlib import pyplot as plt


def flood(image):
	floodfill_image = image.copy()
	h, w = floodfill_image.shape[:2]                                         
	mask = np.zeros((h + 2 , w + 2) , np.uint8)                              
	seeds = []
	for x in xrange(0 , w , 5):
		seeds.append(tuple([0 , x]))
		seeds.append(tuple([h - 5 , x]))
		seeds.append(tuple([x , 0]))
		seeds.append(tuple([x , w - 5]))
	for seed in seeds:
		cv2.floodFill(floodfill_image , mask , seed , 0 , loDiff = 2 , upDiff = 2)
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

#rgb_image = cv2.imread(base.images[0].path , cv2.IMREAD_COLOR)
#hsi_chanels(rgb_image)

#rgb_image = cv2.imread(base.images[0].path)
#rgb_chanels(rgb_image)

for image in base.images:
	print(image.path)
	rgb_image = cv2.imread(image.path)
	grayscale_image = cv2.imread(image.path ,cv2.IMREAD_GRAYSCALE)
	saturation = hsi_chanels(rgb_image)
	blur_image = cv2.GaussianBlur(saturation,(5,5),0)
   	otsu_image = cv2.threshold(blur_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
	flooded_image = flood(otsu_image)
	opening = cv2.morphologyEx(flooded_image, cv2.MORPH_OPEN, np.ones((5,5) , np.uint8))
	img, contours, hierarchy = cv2.findContours(opening.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	if len(contours) > 1:
		bigger_index = 0
		bigger_area = 0
		for contour_index in range(0 , len(contours)):
			area = cv2.contourArea(contours[contour_index])
			if area > bigger_area:
				bigger_area = area
				bigger_index = contour_index
		(x,y),cell_radius = cv2.minEnclosingCircle(contours[bigger_index])
		cell_center = (int(x),int(y))
		cell_radius = int(cell_radius)
		contours.remove(contours[bigger_index])
		for contour in contours:
			(x,y),object_radius = cv2.minEnclosingCircle(contour)
			object_center = (int(x),int(y))
			object_radius = int(object_radius)
			if (object_center[0] > cell_center[0] + cell_radius or object_center[0] < cell_center[0] - cell_radius) and (object_center[1] > cell_center[1] + cell_radius or object_center[1] < cell_center[1] - cell_radius):
				cv2.circle(opening,object_center,object_radius + 10,0,-1)


	cv2.imshow('original' , rgb_image )
	cv2.imshow('closed' , opening )
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


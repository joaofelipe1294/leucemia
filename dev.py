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
		cv2.floodFill(floodfill_image , mask , seed , 255 , loDiff = 2 , upDiff = 2)
	return floodfill_image


def show_histogram(image_path):
	img = cv2.imread(image_path , cv2.IMREAD_COLOR)
	color = ('b','g','r')
	for i,col in enumerate(color):
		histr = cv2.calcHist([img],[i],None,[256],[0,256])
		plt.plot(histr,color = col)
		plt.xlim([0,256])
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
	rgb_image = cv2.imread( image.path)
	saturation = hsi_chanels(rgb_image)
	cv2.imshow('saturation' , saturation)
	cv2.waitKey(0)
	#rgb_chanels(rgb_image)
	#show_histogram(image.path)
	#cv2.imshow('image' ,cv2.imread( image.path))
	#cv2.waitKey(250)
	original_image = cv2.imread( image.path, 0)
	equalized_image = cv2.equalizeHist(original_image)
	flooded_image = flood(original_image)
	laplacian = cv2.Laplacian(flooded_image , cv2.CV_8U)
	edges = cv2.Canny(flooded_image,10,200)
	erosion = cv2.erode(cv2.bitwise_not(edges),np.ones((2,2) , np.uint),iterations = 1)
	dilation = cv2.dilate(flooded_image,np.ones((3,3) , np.uint),iterations = 1)
	closing = cv2.morphologyEx(flooded_image, cv2.MORPH_CLOSE, np.ones((3,3) , np.uint8))
	#show = np.concatenate((original_image , equalized_image & erosion ,flooded_image & erosion , dilation , closing ) , axis=1)
	#cv2.imshow('resault' , show)
	#cv2.waitKey(0)"""



#img = cv2.imread('home.jpg')
#color = ('b','g','r')
#for i,col in enumerate(color):
# histr = cv2.calcHist([img],[i],None,[256],[0,256])
#plt.plot(histr,color = col)
#plt.xlim([0,256])
#plt.show()



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


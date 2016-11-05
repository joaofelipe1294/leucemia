import cv2
import numpy as np
from modules.base.base_loader import BaseLoader


def get_pxs(rgb_image , seed , kernel_size):
	kernel_ray = int(kernel_size / 2)
	for x in xrange(seed[0] - kernel_ray, seed[0] + kernel_ray):
		for y in xrange(seed[1] - kernel_ray , seed[1] + kernel_ray):
			rgb_image.itemset((x , y , 0) , 255)
			rgb_image.itemset((x , y , 1) , 255)
			rgb_image.itemset((x , y , 2) , 255)



base = BaseLoader(train_base_path = 'bases/teste_segmentacao' ,  validation_base_path = 'bases/Teste_ALL_IDB2/ALL')
base.load()
for image in base.train_images:
	rgb_image = cv2.imread(image.path)
	seed = tuple([int(rgb_image.shape[0] / 2) , int(rgb_image.shape[1] / 2)])
	kernel_size = 31
	get_pxs(rgb_image , seed , kernel_size)	
	cv2.imshow('rgb_image' , rgb_image)
	cv2.waitKey(300)
'''
image_path = 'bases/ALL/Im001_1.tif'
rgb_image = cv2.imread(image_path)
seed = tuple([int(rgb_image.shape[0] / 2) , int(rgb_image.shape[1] / 2)])
kernel_size = 31
get_pxs(rgb_image , seed , kernel_size)
cv2.imshow('rgb_image' , rgb_image)
cv2.waitKey(0)
'''
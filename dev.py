from base_loader import BaseLoader
import cv2
import numpy as np



base_loader = BaseLoader()
base_loader.load('ALL_IDB2/img')

""" mostra todas as imagens pre processadas """
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
"""""""""""""""""""""""""""""""""""""""""
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


#print(base_loader.cells[24])
#image_path = base_loader.cells[24].image_path
#gray_scale_image = cv2.imread(image_path , 0)
#print(gray_scale_image.shape)
#cv2.imshow('yo' , gray_scale_image)
#cv2.waitKey(0)
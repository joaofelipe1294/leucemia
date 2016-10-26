import cv2
import numpy as np
from modules.base.base_loader import BaseLoader 
from modules.image_processing.image_chanels import ImageChanels
from modules.image_processing.filters import OtsuThreshold
from modules.image_processing.filters import FloodBorders

base = BaseLoader(train_base_path = 'bases/teste_segmentacao' ,  validation_base_path = 'bases/Teste_ALL_IDB2/ALL')
base.load()
#process(base.train_images[0].path)

file = open('hemacias.csv' , 'w')
for image in base.train_images:
	print(image.path)
	rgb_image = cv2.imread(image.path)
	r , g , b = ImageChanels(rgb_image).rgb(display = False)
	h , s , v = ImageChanels(rgb_image).hsv(display = False)
	s_th = OtsuThreshold(s).process()
	th = OtsuThreshold(b).process()
	sum_img = th + s_th
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
	closing = cv2.morphologyEx(sum_img, cv2.MORPH_CLOSE, kernel)
	not_image = cv2.bitwise_not(closing)
	for x in xrange(0,not_image.shape[0]):
		for y in xrange(0,not_image.shape[1]):
			if not_image.item(x,y) < 200:
				not_image.itemset(x,y,0)
			else:
				not_image.itemset(x,y,1)
	mask = cv2.merge((not_image , not_image , not_image))
	result = rgb_image * mask
	r , g , b = ImageChanels(result).rgb()
	h , s , v = ImageChanels(result).hsv()
	
	for x in xrange(0,result.shape[0]):
		for y in xrange(0,result.shape[1]):
			bv = r.item(x, y)
			gv = g.item(x, y)
			rv = b.item(x, y)
			hv = h.item(x, y)
			sv = s.item(x, y)
			vv = v.item(x, y)
			if bv != 0 or gv != 0 or rv != 0 or hv != 0 or sv != 0 or vv != 0:
				file.write(str(bv) + ',' + str(gv) + ',' + str(rv) + ',' + str(hv) + ',' + str(sv) + ',' + str(vv) + '\n')
	print('processed ' + image.path)
	
	#cv2.imshow('closing' , closing)
	#cv2.imshow('sum' , sum_img)
	#cv2.imshow('sth' , s_th)
	#cv2.imshow('th' , th)
	#cv2.imshow('original' , rgb_image)
	#cv2.imshow('result' , result)
	#cv2.waitKey(300)
file.close()





























































































































































































































































"""for image in base.train_images:
	print(image.path)
	rgb_image = cv2.imread(image.path)
	h , s , v = ImageChanels(rgb_image).hsv()
	sum_img = h + s
	th = OtsuThreshold(sum_img).process()
	
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	dilation = cv2.dilate(th,kernel,iterations = 1)
	#closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
	flooded = FloodBorders(th , value = 125).process()
	cv2.imshow('th' , th)
	cv2.imshow('result' , dilation)
	cv2.waitKey(0)
"""
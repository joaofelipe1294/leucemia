import cv2
import numpy as np
from modules.base.base_loader import BaseLoader 
from modules.image_processing.image_chanels import ImageChanels
from modules.image_processing.filters import OtsuThreshold
from modules.image_processing.filters import FloodBorders
from modules.image_processing.contour import Contour
from modules.image_processing.filters import RegionGrowing



def contour_area(contours):
	if len(contours) == 1:
		return cv2.contourArea(contours[0])
	else:
		total_area = 0
		for contour in contours:
		 	total_area += cv2.contourArea(contour)
		return total_area

			



#base = BaseLoader(train_base_path = 'bases/teste_segmentacao' ,  validation_base_path = 'bases/Teste_ALL_IDB2/ALL')
base = BaseLoader(train_base_path = 'bases/Teste_ALL_IDB2/ALL' ,  validation_base_path = 'bases/Teste_ALL_IDB2/ALL')
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
				not_image.itemset(x,y,1)
			else:
				not_image.itemset(x,y,0)
	mask = cv2.merge((not_image , not_image , not_image))
	result = rgb_image * mask
	h , s , v = ImageChanels(result).hsv(display = False)
	binary_h = OtsuThreshold(h).process()
	binary_s = OtsuThreshold(s).process()
	flooded_h = FloodBorders(binary_h , value = 0).process()
	flooded_s = FloodBorders(binary_s , value = 0).process()
	h_ctns , h_ctns_image = Contour().get_contours(flooded_h)
	s_ctns , s_ctns_image = Contour().get_contours(flooded_s)
	h_rg = RegionGrowing(h_ctns_image , seed = (int(h.shape[0] / 2) , int(h.shape[1] / 2)) , value = 255).process()
	s_rg = RegionGrowing(s_ctns_image , seed = (int(h.shape[0] / 2) , int(h.shape[1] / 2)) , value = 255).process()
	h_opening = cv2.morphologyEx(h_rg, cv2.MORPH_OPEN, kernel)
	s_opening = cv2.morphologyEx(s_rg, cv2.MORPH_OPEN, kernel)
	h_opening = cv2.threshold(h_opening,127,255,cv2.THRESH_BINARY)[1]
	s_opening = cv2.threshold(s_opening,127,255,cv2.THRESH_BINARY)[1]
	h_ctns , h_ctns_image = Contour().get_contours(h_opening)
	s_ctns , s_ctns_image = Contour().get_contours(s_opening)
	h_area = contour_area(h_ctns)
	s_area = contour_area(s_ctns)
	print('H_AREA : ' + str(h_area))
	print('S_AREA : ' + str(s_area))
	new_mask = np.array((0,0))
	if h_area > s_area:
		new_mask = cv2.threshold(h_opening,127,1,cv2.THRESH_BINARY)[1]
	else:
		new_mask = cv2.threshold(s_opening,127,1,cv2.THRESH_BINARY)[1]
	mask = cv2.merge((new_mask , new_mask , new_mask))
	result_processes = rgb_image * mask
	
	cv2.imshow('result' , result_processes)
	cv2.waitKey(300)
	#cv2.imshow('result' , result)
	#cv2.imshow('gray' , gray_image)
	#cv2.imshow('flooded' , flooded_image)
	#cv2.imshow('flooded_h' , flooded_h)
	#cv2.imshow('flooded_s' , flooded_s)
	#cv2.imshow('h_open' , h_ctns_image)
	#cv2.imshow('s_open' , s_ctns_image)
	#cv2.imshow('original' , rgb_image)
	#cv2.imshow('saturation' , binary_s)
	#cv2.imshow('hue' , binary_h)
	#cv2.waitKey(0)


















	"""
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
	print('processed ' + image.path)"""
	
	#cv2.imshow('closing' , closing)
	#cv2.imshow('sum' , sum_img)
	#cv2.imshow('sth' , s_th)
	#cv2.imshow('th' , th)
	#cv2.imshow('original' , rgb_image)
	
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
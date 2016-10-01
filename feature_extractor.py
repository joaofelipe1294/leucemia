import cv2
import numpy as np
import math
from image_chanels import ImageChanels

class FeatureExtractor(object):

	def __init__(self , segmented_image):
		self.segmented_image = segmented_image
		self.area = 0
		self.perimeter = 0
		self.excess = 0
		

	def get_features(self):
		gray_image = cv2.cvtColor(self.segmented_image, cv2.COLOR_BGR2GRAY)
		binary_image = cv2.threshold(gray_image,1,255,cv2.THRESH_BINARY)[1]
		contours_image, contours, hierarchy = cv2.findContours(binary_image.copy() , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
		cv2.drawContours(contours_image, contours, -1, 255, 1)
		self.get_area(gray_image)
		self.get_perimeter(contours)
		self.get_excess(contours)
		self.get_average()
		return self.area , self.perimeter , self.excess , self.average


	def get_area(self , gray_image):
		height , width = gray_image.shape[:2]
		area = np.count_nonzero(gray_image)
		if area:
			normalized_area = (area * 100) / (height * width)
		else:
			normalized_area = 0
		self.area =  normalized_area
		

	def get_perimeter(self , contours):
		cell_perimeter = 0
		for contour in contours:
			cell_perimeter += int(cv2.arcLength(contour , True))
		self.perimeter = cell_perimeter 


	def get_excess(self , contours):
		cell_area = 0
		circunference_area = 0
		for contour in contours:
			cell_area += int(cv2.contourArea(contour))
			(x,y),radius = cv2.minEnclosingCircle(contour)
			radius = int(radius)
			circunference_area += int(math.pi * (radius * radius))
		if cell_area:
			relative_area = (cell_area * 100) / circunference_area
		else:
			relative_area = 0
		self.excess = relative_area
		

	def get_average(self):
		red , green , blue = ImageChanels(self.segmented_image).rgb()
		mean_image = (red + green + blue) / 3
		occurences = 0
		sum_values = 0
		for line in xrange(0 , self.segmented_image.shape[0]):
			for col in xrange(0 , self.segmented_image.shape[1]):
				value = mean_image.item(line , col)
				if value > 0:
					occurences += 1
					sum_values += value
		if sum_values > 0:
			self.average = int(sum_values / occurences)
		else:
			self.average = 0
import cv2
import numpy as np
import math

class FeatureExtractor(object):

	def __init__(self , segmented_image):
		self.segmented_image = segmented_image
		self.area = 0
		self.variance = 0
		self.perimeter = 0
		self.excess = 0

	def get_features(self):
		gray_image = cv2.cvtColor(self.segmented_image, cv2.COLOR_BGR2GRAY)
		binary_image = cv2.threshold(gray_image,1,255,cv2.THRESH_BINARY)[1]
		contours_image, contours, hierarchy = cv2.findContours(binary_image.copy() , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
		cv2.drawContours(contours_image, contours, -1, 255, 1)
		self.get_area(gray_image)
		self.get_variance(gray_image)
		self.get_perimeter(contours)
		self.get_excess(contours)
		return self.area , self.variance , self.perimeter , self.excess


	def get_area(self , gray_image):
		height , width = gray_image.shape[:2]
		area = np.count_nonzero(gray_image)
		normalized_area = (area * 100) / (height * width)
		self.area =  normalized_area
		

	def get_variance(self , gray_image):
		variance = np.var(gray_image)
		self.variance = variance


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
		relative_area = (cell_area * 100) / circunference_area
		self.excess = relative_area
		
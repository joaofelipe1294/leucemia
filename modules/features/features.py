import cv2
import numpy as np
from modules.image_processing.segmentation import SegmentNucleus
from modules.image_processing.filters import OtsuThreshold

class Features(object):

	def __init__(self , segmented_image):
		self.segmented_image = segmented_image
		self.gray_image = cv2.cvtColor(self.segmented_image, cv2.COLOR_BGR2GRAY)
		self.array_image = self.clean_image()


	def clean_image(self):
		reshaped_image = self.gray_image.reshape(-1)
		clean_image = reshaped_image[reshaped_image != 0]
		return clean_image


	def mean(self):
		mean = 0
		if len(self.array_image) > 0:
			mean = np.mean(self.array_image)
		else:
			mean = 0
		return mean

	def median(self):
		median = 0
		if len(self.array_image) > 0:
			median = np.median(self.array_image)
		else:
			median = 0
		return median


	def standard_deviation(self):
		standard_deviation = 0
		if len(self.array_image):
			standard_deviation = np.std(self.array_image)
		else:
			standard_deviation = 0
		return standard_deviation


	def get_nucleus_area(self , original_image):
		nucleus_image = SegmentNucleus(original_image).process()
		contours_image , contours, hierarchy = cv2.findContours(nucleus_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #metodo usado para recuperar os contornos de uma imagem binaria
	 	contours_area = 0
	 	for contour in contours:
	 		contours_area += cv2.contourArea(contour)
	 	return contours_area


 	def get_cell_area(self):
		otsu = OtsuThreshold(self.gray_image).process()
		contours_image , contours, hierarchy = cv2.findContours(otsu.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #metodo usado para recuperar os contornos de uma imagem binaria
		contours_area = 0
		for contour in  contours:
			contours_area += cv2.contourArea(contour)
		return contours_area


	def nucleus_proportion(self , original_image):
		nucleus_area = self.get_nucleus_area(original_image)
		cell_area = self.get_cell_area()
		if cell_area < nucleus_area:
			cell_area = nucleus_area + 1000
		proportion = (nucleus_area  * 100) / cell_area #pega a porcentagem
		return proportion
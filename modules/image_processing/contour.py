import cv2 
import numpy as np


class Contour(object):

	def get_contours(self , image):
		contours_image , contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cv2.drawContours(contours_image, contours, -1,255, 1)
		#self.contours = contours
		return contours , contours_image
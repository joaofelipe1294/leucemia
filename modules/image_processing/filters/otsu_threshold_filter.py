import cv2
import numpy as np


class OtsuThresholdFilter(object):

	def process (self , image):
		#aplica o threshold baseado na tecnica de OTSU
		blur_image = cv2.GaussianBlur(image , (5,5) , 0)                                            #borra a imagem aplicando um filtro gaussiano , necessario para que o threshold OTSU funcione
	   	otsu_image = cv2.threshold(blur_image , 0 , 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]   #aplica o threshold de otsu na imagem borrada
	   	return otsu_image
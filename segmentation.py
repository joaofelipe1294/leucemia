import cv2
import numpy as np
from image_chanels import ImageChanels

class Segmentation(object):


	def __init__(self , image_path):
		self.rgb_image = cv2.imread(image_path)


	def process(self):                                      
		#faz a segmentacao da celula de interece   
		saturation = ImageChanels(self.rgb_image).hsi('S')  #extraido canal relativo a Saturacao
		threshold_image = self.otsu_threshold(saturation)
		return threshold_image


	def otsu_threshold(self , image):
		#aplica o threshold baseado na tecnica de OTSU
		blur_image = cv2.GaussianBlur(image , (5,5) , 0)                                     #borra a imagem aplicando um filtro gaussiano , necessario para que o threshold OTSU funcione
	   	otsu_image = cv2.threshold(blur_image , 0 , 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]   #aplica o threshold de otsu na imagem borrada
	   	return otsu_image

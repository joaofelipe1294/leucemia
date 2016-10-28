import numpy as np
import cv2
import numpy as np


class ImageChanels(object):


	def __init__(self , rgb_image):
		self.rgb_image = rgb_image
		self.height = self.rgb_image.shape[0]
		self.width = self.rgb_image.shape[1]


	def rgb(self , chanel = None , display = False):
		#trabalha com canais RGB
		if chanel:                       #verifica se deve retornar um canal especifico 
			if chanel == 'B':
				chanel_index = 0
			elif chanel == 'G':
				chanel_index = 1
			elif chanel == 'R':
				chanel_index = 2
			color_chanel = cv2.split(self.rgb_image)[chanel_index]
			if display:
				cv2.imshow('chanel' , color_chanel)
				cv2.waitKey(0)
			return color_chanel
		else:
			blue_chanel , green_chanel , red_chanel = cv2.split(self.rgb_image)
			if display:
				show = np.concatenate(( red_chanel , green_chanel , blue_chanel) , axis=1)
				cv2.imshow('resault' , show)
				cv2.waitKey(0)				
			return red_chanel , green_chanel , blue_chanel


	def hsv(self, chanel = None , display = False):
		#trabalha com canais HSV
		hsv_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)
		if chanel:                           #trabalha com apenas um canal
			if chanel == 'H':
				chanel_index = 0
			elif chanel == 'S':
				chanel_index = 1
			elif chanel == 'V':
				chanel_index = 2
			image_chanel = cv2.split(hsv_image)[chanel_index]
			if display:
				cv2.imshow('image_chanel' , image_chanel)
				cv2.waitKey(0)
			return image_chanel
		else:
			h , s , v = cv2.split(hsv_image)
			if display:
				show = np.concatenate(( h , s , v) , axis=1)
				cv2.imshow('resault' , show)
				cv2.waitKey(0)
			return h , s , v		

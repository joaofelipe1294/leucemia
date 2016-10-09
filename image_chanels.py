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
			return color_chanels
		else:
			blue_chanel , green_chanel , red_chanel = cv2.split(self.rgb_image)
			if display:
				show = np.concatenate(( red_chanel , green_chanel , blue_chanel) , axis=1)
				cv2.imshow('resault' , show)
				cv2.waitKey(0)				
			return red_chanel , green_chanel , blue_chanel


	def hsv(self, chanel = None , display = False):
		#trabalha com canais HSI
		chanel_index = -1
		if chanel:                           #trabalha com apenas um canal
			if chanel == 'H':
				chanel_index = 0
			elif chanel == 'S':
				chanel_index = 1
			elif chanel_index == 'V':
				chanel = 2
			hsi_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)
			chanel_image = np.zeros((self.height , self.width) , np.uint8)
			for line in xrange(0 , self.height):
				for col in xrange(0 , self.width):
					val = hsi_image.item( line, col, chanel_index)
					chanel_image.itemset((line , col), val)
			if display:
				cv2.imshow('chanel' , chanel_image)
				cv2.waitKey(0)
			return chanel_image
		else:                               #trabalha com todos os canais
			hsi_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)
			hue_chanel = np.zeros((self.height , self.width) , np.uint8)
			saturation_chanel = np.zeros((self.height , self.width) , np.uint8)
			intensity_chanel = np.zeros((self.height , self.width) , np.uint8)
			chanels = [hue_chanel , saturation_chanel , intensity_chanel]
			for chanel_index in range(0,3):
				for line in xrange(0,self.height):
					for col in xrange(0,self.width):
						value = hsi_image.item(line , col , chanel_index)
						chanels[chanel_index].itemset((line , col) , value)
			if display:
				show = np.concatenate(( hue_chanel , saturation_chanel , intensity_chanel) , axis=1)
				cv2.imshow('resault' , show)
				cv2.waitKey(0)		
			return 	hue_chanel , saturation_chanel , intensity_chanel



image = cv2.imread('ALL_IDB2/img/Im001_1.tif')
#cv2.imshow('image' , image)
#cv2.waitKey(0)
r , g , b = ImageChanels(image).rgb(display = True)
show = np.concatenate(( r , g , b) , axis=1)
cv2.imshow('resault' , show)
cv2.waitKey(0)				
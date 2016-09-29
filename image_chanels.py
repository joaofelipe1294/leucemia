import numpy as np
import cv2
import numpy as np


class ImageChanels(object):


	def __init__(self , rgb_image):
		self.rgb_image = rgb_image
		self.height = self.rgb_image.shape[0]
		self.width = self.rgb_image.shape[1]


	def rgb(self , chanel = None , display = False):
		chanel_index = -1                    #inicializado com o valor -1 pelo fato de que esta variavel deve ser do tipo int
		if chanel:
			if chanel == 'R':
				chenel_index = 2
			elif chanel == 'G':
				chanel_index = 1
			elif chanel == 'B':
				chanel_index = 0
			chanel_image = np.zeros((self.height , self.width) , np.uint8)
			for line in xrange(0,self.height):
				for col in xrange(0,self.width):
					value = self.rgb_image.item(line , col , chanel_index)
					chanel_image.itemset((line , col) , value)
			if display:
				cv2.imshow('chanel' , chanel_image)
				cv2.waitKey(0)
			return chanel_image
		else:
			red_chanel = np.zeros((self.height , self.width) , np.uint8)
			green_chanel = np.zeros((self.height , self.width) , np.uint8)
			blue_chanel = np.zeros((self.height , self.width) , np.uint8)
			color_chanels = [blue_chanel , green_chanel , red_chanel]
			for chanel_index in range(0,3):
				for line in xrange(0,self.height):
					for col in xrange(0,self.width):
						value = self.rgb_image.item(line , col , chanel_index)
						color_chanels[chanel_index].itemset((line , col) , value)
			if display:
				show = np.concatenate(( red_chanel , green_chanel , blue_chanel) , axis=1)
				cv2.imshow('resault' , show)
				cv2.waitKey(0)		

			return 	red_chanel , green_chanel , blue_chanel


		#height , width = rgb_image.shape[:2]
		#red_image = np.zeros((height , width) , np.uint8)
		#green_image = np.zeros((height , width) , np.uint8)
		#blue_image = np.zeros((height , width) , np.uint8)
		#color_chanels = [blue_image , green_image , red_image]
		#for chanel_index in range(0,3):
		#	for line in xrange(0,height):
		#		for col in xrange(0,width):
		#			value = rgb_image.item(line , col , chanel_index)
		#			color_chanels[chanel_index].itemset((line , col) , value)
		#show = np.concatenate(( red_image , green_image , blue_image) , axis=1)
		#cv2.imshow('resault' , show)
		#cv2.imshow('rgb' , rgb_image)
		#cv2.waitKey(0)


	def hsi(self, chanel_return = 'S'):
		if chanel_return == 'H':
			chanel_index = 0
		elif chanel_return == 'S':
			chanel_index = 1
		elif chanel_return == 'I':
			chanel_index = 2
		hsi_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
		height , width = rgb_image.shape[:2]
		chanel = np.zeros((height , width) , np.uint8)
		for line in xrange(0 , height):
			for col in xrange(0 , width):
				val = hsi_image.item( line, col, chanel_index)
				chanel.itemset((line , col), val)
		return chanel
from matplotlib import pyplot as plt
import cv2

class Histogram(object):
	
	def rgb(self , rgb_image):
		#exibe o histograma de uma imagem RGB
		color = ('b','g','r')
		for i,col in enumerate(color):
			histr = cv2.calcHist([rgb_image],[i],None,[256],[0,256])
			plt.plot(histr,color = col)
			plt.xlim([0,256])
		plt.show()


	def gray_scale(self , grayscale_image):
		#exibe o histograma de uma imagem em tons de cinza
		plt.hist(grayscale_image.ravel(),256,[0,256])
		plt.show()
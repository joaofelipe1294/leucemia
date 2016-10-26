import cv2
import numpy as np
from abc import ABCMeta , abstractmethod


##################################################################################################################


class Filter(object):
	"""	
		classe usada como base para todas as demais classes que servem para aplicar um filtro em uma imagem , 
		os atributos particulares de cada filtro devem ser recebidos no contrutor das classes que herdam de 
		Filter	
	"""

	__metaclass__ = ABCMeta  #usado para tornas a classe filter abstrata

	def __init__(self , image):
		"""
			parametros 
			   	@image - imagem em tons de cinza ou binaria que sera processada pelo filtro
			atributos
				@image  - matriz numpy referente a imagem em tons de cinza recebida no construtor
				@height - altura da imagem
				@width  - largura da imagem
		"""
		self.image = image
		self.height = self.image.shape[0]
		self.width = self.image.shape[1]

	@abstractmethod
	def process(self): 
		"""
			define o metodo abstrato que todas as classes filha devem implementar, esse metodo deve retornar a
		 	imagem resultante da aplicacao do filtro  
		"""
		pass	


#################################################################################################################


#################################################################################################################


class OtsuThreshold(Filter):
	#classe que aplica o threshold de Otsu
	

	def __init__(self , image):
		super(OtsuThreshold , self).__init__(image)

	def process(self):
		blur_image = cv2.GaussianBlur(self.image , (5,5) , 0)                                       #borra a imagem aplicando um filtro gaussiano , necessario para que o threshold OTSU funcione
	   	otsu_image = cv2.threshold(blur_image , 0 , 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]   #aplica o threshold de otsu na imagem borrada
	   	return otsu_image


#################################################################################################################

imagem = np.ones((5,5) , np.uint8)
OtsuThreshold(imagem).process()



import cv2
import numpy as np
from abc import ABCMeta , abstractmethod


##################################################################################################################


class Segmentation(object):
	"""
		classe abstrata usada para criar um padrao no desenvolvimento das classes que irao implementar os 
		pipelines de segmentacao , todas as classes com o objetivo de segmentar uma imagem devem herdar 
		de Segmentation e implementar o pipeline de segmentacao no metodo abstrato process()
	"""


	__metaclass__ = ABCMeta  #usado para tornas a classe filter abstrata

	def __init__(self , image_path = ''):
		"""
			parametros
				@image_path - string que contem o caminho da imagem que sera segmentada
			atributos
				@image_path - string que contem o caminho da imagem que sera segmentada
				@rgb_image  - imagem rgb lida a partir do caminho informado 
		"""
		self.image_path = image_path
		self.rgb_image = cv2.imread(self.image_path)


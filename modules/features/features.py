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


############################################################################################

############################################################################################


class MinMax(object):
	"""
		classe usada para normalizacao de dados , implementa o algoritmo de normalizacao 
		MinMax
	"""

	def __init__(self , values):
		"""
			parametros
				@values - uma lista que contem outras listas com os valores referentes
						  referentes ao vetor de caracteristicas de cada objeto
		  	atributos 
		  		@values - array numpy com todos os seus valores do tipo float
		"""

		values = [[float(j) for j in column] for column in values] #converte os valores para float
		self.values = np.array(values) #converte uma list de lists em um array numpy


	def normalize(self , minimum = 0, maximum = 1):
		"""
			Metodo que aplica o algoritmo de normalizacao MinMax
			parametros
				@minimum - valor referente a constante min da formula MinMax
				@maximum - valor referente a constante max da formula MinMax
		"""

		normalized_values = np.zeros((self.values.shape[0] , self.values.shape[1] - 1)) #cria um array numpy composto apenas por zeros , esse vetor tera seus valores alterados para os valores normalizados (lembra o hiperplano do Hough) , possui uma coluna a menos do que o original porque as labels sao do tipo int
		for column in xrange(0,self.values.shape[1] - 1) : #loop que roda em cima de uma coluna por vez
			column_values = self.values[:,column] #seleciona apenas uma coluna do array numpy que contem os valores originais
			bigger = max(column_values) #pega o maior valor da coluna
			lower = min(column_values)  #pega o menor valor da coluna
			line = 0
			for value in column_values: #loop que roda em cima de cada linha (elemento) de uma coluna extraida dos valores originais
				normalized_value = 0
				if value != 0:
					normalized_value = minimum + ((value - lower) / (bigger - lower)) * (maximum - minimum) #aplica a formula MinMax
					normalized_values.itemset(line , column , normalized_value) #seta o valor normalizado na matriz normalized_values em sua posicao correta
				line += 1
		labels = self.values[: , self.values.shape[1] - 1] #extrai a ultima coluna dos valores originais (contem apenas as labels)
		labels = [[label] for label in labels] #converte a list labels em uma lista que contem uma label dentro de uma list
		normalized_values = np.append(normalized_values , np.array(labels) , axis=1) #converte a list de lists labels em um array numpy , adiciona essa nova coluna ao fim da matriz normalized_values
		return normalized_values



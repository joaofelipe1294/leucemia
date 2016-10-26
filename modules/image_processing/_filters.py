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


	def valid_resut(self , floodfill_image):
		"""
			metodo usado para validar a saida de alguns filtros , aplicado inicialmente apos a execucao de filtros
			floodFill
		"""
		white = 0 
		for x in xrange(0 , self.height):                              #verifica o numero de px com o valor 255 (branco) apos a aplicacao da funcao cv2.floodFill(), usado futuramente para verificar se houve um erro ou nao 
			for y in xrange(0 , self.width):
				if floodfill_image.item(x,y) == 255:
					white += 1
		if white > ((self.height * self.width * 80) / 100):            #verifica se mais de 80% da imagem ficou branca, o que claramente indica um erro tendo em vista que nenhuma celula ocupa esse tamanho na imagem, caso isso seja verdade invertese a imagem 
			floodfill_image = cv2.bitwise_not(floodfill_image)
		return floodfill_image	


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


#################################################################################################################


class RegionGrowing(Filter):
	"""
		classe que tem como objetivo a aplicacao do filtro de crescimento de regioes , por debaixo dos 
		panos usa o filtro FloodFill presente no openCV , tem como objetivo crescer uma regiao em par-
		ticular
	"""


	def __init__(self , image , seed = None , value = 0):
		"""
			parametros
				@value - valor que sera aplicado nos pixels com valor proximo ao do ponto semente
				@seed  - coordenada do ponto que sera inundado, uma tupla com os valores do eixo X
				         e Y
		"""
		if not seed:
			raise ValueError("ponto semente nao foi informado !!!")
		super(RegionGrowing , self).__init__(image)
		self.value = value
		self.seed = seed
		self.mask = np.zeros((self.height + 2 , self.width + 2) , np.uint8) #cria a mascara exigida pela funcao cv2.floodFill()


	def process(self):
		#aplica o filtro flood_fill
		floodfill_image = self.image.copy()                                                          #cria uma copia da imagem recebida para que a mesma nao seja alterada, uma vez que a funcao cv2.floodFill() altera a imagem recebida como parametro ao inves de retornar uma nova imagem com o filtro aplicado
		cv2.floodFill(floodfill_image , self.mask , self.seed , self.value , loDiff = 2 , upDiff = 2)
		return self.valid_resut(floodfill_image)


#################################################################################################################


#################################################################################################################


class FloodBorders(Filter):
	"""
		classe que aplica o flitro cv2.floodFill() nas bordas da imagem , usado inicalmente para remover ruidos
		das extremidades da imagem
	"""

	def __init__(self , image , value = 0):
		"""
			parametros
				@value - valor que sera aplicado aos pixels com valor proximo ao dos pontos semente
		"""
		super(FloodBorders , self).__init__(image)
		self.value = value
		self.mask = np.zeros((self.height + 2 , self.width + 2) , np.uint8) #cria a mascara exigida pela funcao cv2.floodFill()


	def process(self):
		floodfill_image = self.image.copy()                                                          #cria uma copia da imagem recebida para que a mesma nao seja alterada, uma vez que a funcao cv2.floodFill() altera a imagem recebida como parametro ao inves de retornar uma nova imagem com o filtro aplicado
		seeds = []                                                                          #lista com todas as sementes
		if self.width == self.height:      
			for x in xrange(0 , self.width , 1):                                                #definindo as coordenadas das sementes 
				seeds.append(tuple([0 , x]))
				seeds.append(tuple([self.height - 1 , x]))
				seeds.append(tuple([x , 0]))
				seeds.append(tuple([x , self.width - 1]))
		else:
			limit = 0
			if self.width > self.height:
				limit = self.height
			else:
				limit = self.width
			for x in xrange(0 , limit , 1):
				seeds.append(tuple([0 , x]))
				seeds.append(tuple([self.width - 1 , x]))
				seeds.append(tuple([x , 0]))
				seeds.append(tuple([x , self.height - 1]))	
		for seed in seeds:
			cv2.floodFill(floodfill_image , self.mask , seed , self.value , loDiff = 2 , upDiff = 2)  #aplica a funcao cv2.floodFill() para cada semente criada
		return self.valid_resut(floodfill_image)
	

#################################################################################################################
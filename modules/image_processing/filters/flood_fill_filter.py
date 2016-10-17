import cv2
import numpy as np

class FloodFillFilter(object):


	def __init__(self , image):
		self.image = image
		self.height = image.shape[0]
		self.width = image.shape[1]
		self.mask = np.zeros((self.height + 2 , self.width + 2) , np.uint8) #cria a mascara exigida pela funcao cv2.floodFill()

	def flood_borders(self , value = 0):
		#aplica o filtro flood_fill
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
			cv2.floodFill(floodfill_image , self.mask , seed , value , loDiff = 2 , upDiff = 2)  #aplica a funcao cv2.floodFill() para cada semente criada
		return self.valid_resut(floodfill_image)
	

	def flood_region(self , seed , value = 0):
		#aplica o filtro flood_fill
		floodfill_image = self.image.copy()                                                          #cria uma copia da imagem recebida para que a mesma nao seja alterada, uma vez que a funcao cv2.floodFill() altera a imagem recebida como parametro ao inves de retornar uma nova imagem com o filtro aplicado
		cv2.floodFill(floodfill_image , self.mask , seed , value , loDiff = 2 , upDiff = 2)
		return self.valid_resut(floodfill_image)


	def valid_resut(self , floodfill_image):
		white = 0 
		for x in xrange(0 , self.image.shape[0]):                                                      #verifica o numero de px com o valor 255 (branco) apos a aplicacao da funcao cv2.floodFill(), usado futuramente para verificar se houve um erro ou nao 
			for y in xrange(0 , self.image.shape[1]):
				if floodfill_image.item(x,y) == 255:
					white += 1
		if white > ((self.image.shape[0] * self.image.shape[1] * 80) / 100):                              #verifica se mais de 80% da imagem ficou branca, o que claramente indica um erro tendo em vista que nenhuma celula ocupa esse tamanho na imagem, caso isso seja verdade invertese a imagem 
			floodfill_image = cv2.bitwise_not(floodfill_image)
		return floodfill_image		
import cv2
import numpy as np
from image_chanels import ImageChanels

class Segmentation(object):


	def __init__(self , image_path):
		self.rgb_image = cv2.imread(image_path)
		self.height = self.rgb_image.shape[0]
		self.width = self.rgb_image.shape[1]


	def process(self):                                      
		#faz a segmentacao da celula de interece   
		saturation = ImageChanels(self.rgb_image).hsi('S')  #extraido canal relativo a Saturacao
		threshold_image = self.otsu_threshold(saturation)   #aplica threshold de OTSU no canal referente a saturacao 
		flooded_image = self.flood(threshold_image)         #aplica o filtro flood_fill com o objetivo de remover os objetos colados as extremidades
		opening = cv2.morphologyEx(flooded_image, cv2.MORPH_OPEN, np.ones((5,5) , np.uint8))  #aplica operacao morfologica de abertura para remover pequenos pontos brancos (ruidos) presentes na imagem resultante da operacao anterior 
		return opening


	def otsu_threshold(self , image):
		#aplica o threshold baseado na tecnica de OTSU
		blur_image = cv2.GaussianBlur(image , (5,5) , 0)                                            #borra a imagem aplicando um filtro gaussiano , necessario para que o threshold OTSU funcione
	   	otsu_image = cv2.threshold(blur_image , 0 , 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]   #aplica o threshold de otsu na imagem borrada
	   	return otsu_image


   	def flood(self , image , value = 0 , single_seed = None):
   		#aplica o filtro flood_fill
		floodfill_image = image.copy()                                       #cria uma copia da imagem recebida para que a mesma nao seja alterada, uma vez que a funcao cv2.floodFill() altera a imagem recebida como parametro ao inves de retornar uma nova imagem com o filtro aplicado
		mask = np.zeros((self.height + 2 , self.width + 2) , np.uint8)       #cria a mascara exigida pela funcao cv2.floodFill()                      
		if single_seed == None:                                              #coloca sementes nas extremidades da imagem
			seeds = []                                                       #lista com todas as sementes
			for x in xrange(0 , self.width , 5):                             #definindo as coordenadas das sementes 
				seeds.append(tuple([0 , x]))
				seeds.append(tuple([self.height - 5 , x]))
				seeds.append(tuple([x , 0]))
				seeds.append(tuple([x , self.width - 5]))
			for seed in seeds:
				cv2.floodFill(floodfill_image , mask , seed , value , loDiff = 2 , upDiff = 2) #aplica a funcao cv2.floodFill() para cada semente criada
		else:
			seed = single_seed
			cv2.floodFill(floodfill_image , mask , seed , value , loDiff = 2 , upDiff = 2)
		white = 0 
		for x in xrange(0,image.shape[0]):                                   #verifica o numero de px com o valor 255 (branco) apos a aplicacao da funcao cv2.floodFill(), usado futuramente para verificar se houve um erro ou nao 
			for y in xrange(0,image.shape[1]):
				if floodfill_image.item(x,y) == 255:
					white += 1
		if white > ((image.shape[0] * image.shape[1] * 80) / 100):           #verifica se mais de 80% da imagem ficou branca, o que claramente indica um erro tendo em vista que nenhuma celula ocupa esse tamanho na imagem, caso isso seja verdade invertese a imagem 
			floodfill_image = cv2.bitwise_not(floodfill_image)
		return floodfill_image
import cv2
import numpy as np
from image_chanels import ImageChanels
import math


class Segmentation(object):


	def __init__(self , image_path):
		self.rgb_image = cv2.imread(image_path)
		self.height = self.rgb_image.shape[0]
		self.width = self.rgb_image.shape[1]
		self.contours = []
		self.cell_center = 0
		self.cell_radius = 0
		self.mask = None


	def process(self):                                      
		#faz a segmentacao da celula de interece   
		saturation = ImageChanels(self.rgb_image).hsv('S')                                         #extraido canal relativo a Saturacao
		threshold_image = self.otsu_threshold(saturation)                                          #aplica threshold de OTSU no canal referente a saturacao 
		flooded_image = self.flood(threshold_image)                                                #aplica o filtro flood_fill com o objetivo de remover os objetos colados as extremidades
		opened_image = cv2.morphologyEx(flooded_image, cv2.MORPH_OPEN, np.ones((5,5) , np.uint8))  #aplica operacao morfologica de abertura para remover pequenos pontos brancos (ruidos) presentes na imagem resultante da operacao anterior 
		contour_image = self.get_contours(flooded_image)                                           #computa uma imagem com os contornos desenhados e uma lista com aas coordenadas dos contornos 
		self.cell_center , self.cell_radius = self.find_interest_cell()                            #computa o ponto central e o raio da celula de interesse 
		if len(self.contours) == 0:                                                                     #se o numero de contornos for igual a zero significa que existe apenas um objeto na imagem opened_image logo a mascara ja esta correta
			self.mask = opened_image
		else:
			self.mask = self.remove_noise_objects(contour_image , threshold_image)
		self.mask = self.build_mask()
		return self.mask


	def otsu_threshold(self , image):
		#aplica o threshold baseado na tecnica de OTSU
		blur_image = cv2.GaussianBlur(image , (5,5) , 0)                                            #borra a imagem aplicando um filtro gaussiano , necessario para que o threshold OTSU funcione
	   	otsu_image = cv2.threshold(blur_image , 0 , 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]   #aplica o threshold de otsu na imagem borrada
	   	return otsu_image


   	def flood(self , image , value = 0 , single_seed = None):
   		#aplica o filtro flood_fill
		floodfill_image = image.copy()                                                          #cria uma copia da imagem recebida para que a mesma nao seja alterada, uma vez que a funcao cv2.floodFill() altera a imagem recebida como parametro ao inves de retornar uma nova imagem com o filtro aplicado
		mask = np.zeros((self.height + 2 , self.width + 2) , np.uint8)                          #cria a mascara exigida pela funcao cv2.floodFill()                      
		if single_seed == None:                                                                 #coloca sementes nas extremidades da imagem
			seeds = []                                                                          #lista com todas as sementes
			if self.width == self.height:      
				for x in xrange(0 , self.width , 5):                                                #definindo as coordenadas das sementes 
					seeds.append(tuple([0 , x]))
					seeds.append(tuple([self.height - 5 , x]))
					seeds.append(tuple([x , 0]))
					seeds.append(tuple([x , self.width - 5]))
			else:
				limit = 0
				if self.width > self.height:
					limit = self.height
				else:
					limit = self.width
				for x in xrange(0 , limit , 5):
					seeds.append(tuple([0 , x]))
					seeds.append(tuple([self.width - 5 , x]))
					seeds.append(tuple([x , 0]))
					seeds.append(tuple([x , self.height - 5]))	
			for seed in seeds:
				cv2.floodFill(floodfill_image , mask , seed , value , loDiff = 2 , upDiff = 2)  #aplica a funcao cv2.floodFill() para cada semente criada
		else:
			seed = single_seed
			cv2.floodFill(floodfill_image , mask , seed , value , loDiff = 2 , upDiff = 2)
		white = 0 
		for x in xrange(0,image.shape[0]):                                                      #verifica o numero de px com o valor 255 (branco) apos a aplicacao da funcao cv2.floodFill(), usado futuramente para verificar se houve um erro ou nao 
			for y in xrange(0,image.shape[1]):
				if floodfill_image.item(x,y) == 255:
					white += 1
		if white > ((image.shape[0] * image.shape[1] * 80) / 100):                              #verifica se mais de 80% da imagem ficou branca, o que claramente indica um erro tendo em vista que nenhuma celula ocupa esse tamanho na imagem, caso isso seja verdade invertese a imagem 
			floodfill_image = cv2.bitwise_not(floodfill_image)
		return floodfill_image


	def get_contours(self , image):
		contours_image , contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cv2.drawContours(contours_image, contours, -1,255, 1)
		self.contours = contours
		return contours_image


	def find_interest_cell(self):
		#recebe os contornos de uma imagem e com base neles retorna o ponto central e o raio da celula de interesse , alem de uma lista com os contornos referente apenas a objetos que nao sejam a celula de interesse. A celula de interesse eh aquela que possui uma menor distancia Euclidiana de seu ponto central em relacao ao ponto central da imagem
		image_center_point = tuple([int(self.height / 2) , int(self.width) / 2])               #descobre o ponto central da imagem
		lowest_index = 0
		lowest_distance = None
		for contour_index in range(0 , len(self.contours)):                                         #itera sobre todos os contornos   
			(x,y) , object_radius = cv2.minEnclosingCircle(self.contours[contour_index])            #aplica a funcao cv2.minEnclosingCircle() que recebe um contorno e reorna o raio e o ponto central para a menor circunferencia  possivel que englobe o objeto referente ao contorno passado como parametro
			distance_to_center = math.sqrt(math.pow((x - image_center_point[1]) , 2) + math.pow(y - image_center_point[1] , 2))   #alcula a distancia Euclidiana para o ponto central do contorno em relacao ao ponto central da imagem , tendo em vista que a celula de interece eh a celula mais proxima ao centro da imagem 
			if lowest_distance == None or distance_to_center < lowest_distance:                #verifia se a distancia do ponto central do contorno e o ponto central da imagem eh menor do que a menor distancia computada anteriormente 
				lowest_index = contour_index
				lowest_distance = distance_to_center
		(x,y),cell_radius = cv2.minEnclosingCircle(self.contours[lowest_index])                     #obtem o ponto central e o raio da menor circunferencia possivel que engloba a celula mais proxima do centro da imagem (celula de interesse)
		cell_radius = int(cell_radius)                                                         #normaliza o tipo de dado referente ao raio da "celula"
		cell_center = (int(x),int(y))                                                          #normaliza o tipo de dado referente ao ponto central da celula
		self.contours.pop(lowest_index)                                                             #remove o contorno da celula de interesse da lista que contem os contornos da imagem 
		return cell_center , cell_radius


	def remove_noise_objects(self , contours_image , threshold_image):
		#metoco com o objetivo de remover qualquer objeto da imagem que nao seja a celula de interesse
		flooded_image = self.flood(contours_image.copy() , 255 , single_seed = self.cell_center)   #preenche a celula central 
		for contour in self.contours:                                                              #varre todos os contornos e verifica se a celula possui o nucleo vazado , nesse caso ocorre uma excecao que sera corrigida mais pra frente
			(x,y) , object_radius = cv2.minEnclosingCircle(contour)                                #computa o raio e o ponto central do contorno
			object_radius = int(object_radius)                                                     #normaliza o tipo de dado  
			object_center = (int(x),int(y))                                                        #normaliza o tipo de dado
			if (object_center[0] + object_radius > self.cell_center[0] + self.cell_radius and object_center[0] - object_radius < self.cell_center[0] - self.cell_radius) and (object_center[1] + object_radius > self.cell_center[1] + self.cell_radius and object_center[1] - object_radius < self.cell_center[1] + self.cell_radius): #verifica se o nucleo real da celula esta em volta do que foi marcado como nucleo
				opened_image = cv2.morphologyEx(threshold_image , cv2.MORPH_OPEN, np.ones((5,5) , np.uint8))
				return opened_image                                                                #nesse caso excepcional a imagem resultante do threshold de OTSU ja esta correta , eh aplicada uma operacao morfologica de abertura para remover pontos de ruido 
		opened_image = cv2.morphologyEx(flooded_image.copy(), cv2.MORPH_OPEN, np.ones((5,5) , np.uint8))   #remove os contornos de objetos que nao sejam a celula central
		return opened_image


	def build_mask(self):
		for line in xrange(0 , self.height):
			for col in xrange(0 , self.width):
				if self.mask.item(line , col) > 200:
					self.mask.itemset((line , col) , 255)
				else:
					self.mask.itemset((line , col) , 0)
		for line in xrange(0 , self.height):
			for col in xrange(0 , self.width):
				if self.mask.item(line , col) == 255:
					self.mask.itemset((line , col) , 1)
		red , green , blue = ImageChanels(self.rgb_image).rgb()
		red = red * self.mask
		green = green * self.mask
		blue = blue * self.mask
		chanels = [blue , green , red]
		copy = self.rgb_image.copy()
		for chanel_index in xrange(0,3):
			for line in xrange(0 , self.height):
				for col in xrange(0 , self.width):
					copy.itemset((line , col , chanel_index) , chanels[chanel_index].item(line , col))
		return copy


"""
	def segment_cell(self):
		gray_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)
		flood = self.flood(gray_image)
		cv2.imshow('gray image' , flood)
		cv2.waitKey(0)		


from base_loader import *




paths = os.listdir('teste')
paths.sort()
images = []
for image_path in paths:
	segmented_image = Segmentation('ALL_IDB2/img/' + image_path).segment_cell()
	#cv2.imshow('segmented image' , segmented_image)
	#cv2.waitKey(350)

#image_path = "ALL_IDB2/img/Im021_1.tif"
#segmented_image = Segmentation(image_path).segment_cell()
#cv2.imshow('segmented image' , segmented_image)
#cv2.waitKey(0)
"""
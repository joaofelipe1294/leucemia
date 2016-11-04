import cv2
import math
import numpy as np
from modules.base.base_loader import BaseLoader
from modules.image_processing.image_chanels import ImageChanels
from modules.image_processing.filters import *


def get_erythrocytes(rgb_image):
		"""
			extrai as hemacias de uma imagem RGB , o retorno eh uma imagem preta com apenas as hemacias em branco 
			parametros
				@rgb_image - imagem rgb que sera processada
		"""
		blue_chanel = ImageChanels(rgb_image).rgb(chanel = 'B')         #separa canais RGB
		saturation = ImageChanels(rgb_image).hsv(chanel ='S')           #separa canais HSV
		saturation_binary = OtsuThreshold(saturation).process()         #aplica threshold na saturacao
		blue_chanel_binary = OtsuThreshold(blue_chanel).process()       #aplica threshold no canal B (azul)
		sum_image = blue_chanel_binary + saturation_binary              #soma os threshold da saturacao ao threshold do canal azul para remover a celula central da imagem , mantem apenas as hemacias em preto e o fundo branco
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))   #kernel circular usado para operacao de abertura     
		closing = cv2.morphologyEx(sum_image, cv2.MORPH_CLOSE, kernel)  #aplica fechamento para remover ruidos (alguns contornos da celila principal podem ter ficado na imagem resultante da som)
		result_image = cv2.bitwise_not(closing)                         #inverte a imgem resultante do fechamento , deixa apenas as hemacias em branco e o resto em preto
		return result_image



def find_interest_cell(contours , image):
		"""
			recebe os contornos de uma imagem e com base neles retorna o ponto central e o raio da celula de interesse
			, alem de uma lista com os contornos referente apenas a objetos que nao sejam a celula de interesse. A 
			celula de interesse eh aquela que possui uma menor distancia Euclidiana de seu ponto central em relacao 
			ao ponto central da imagem
			parametros
				@contours - lista com os contornos de uma imagem
		"""
		image_center_point = tuple([int(image.shape[0] / 2) , int(image.shape[1]) / 2])   #descobre o ponto central da imagem
		lowest_index = 0              #indice da posicao com menor distancia euclidiana do ponto central de um contorno com o centro da imagem
		lowest_distance = None        #valor da menor distancia do ponto central de um contorno em relacao ao ponto central da imagem
		for contour_index in range(0 , len(contours)):                                   #itera sobre todos os contornos   
			(x,y) , object_radius = cv2.minEnclosingCircle(contours[contour_index])      #aplica a funcao cv2.minEnclosingCircle() que recebe um contorno e reorna o raio e o ponto central para a menor circunferencia  possivel que englobe o objeto referente ao contorno passado como parametro
			distance_to_center = math.sqrt(math.pow((x - image_center_point[1]) , 2) + math.pow(y - image_center_point[1] , 2))   #alcula a distancia Euclidiana para o ponto central do contorno em relacao ao ponto central da imagem , tendo em vista que a celula de interece eh a celula mais proxima ao centro da imagem 
			if lowest_distance == None or distance_to_center < lowest_distance:                #verifia se a distancia do ponto central do contorno e o ponto central da imagem eh menor do que a menor distancia computada anteriormente 
				lowest_index = contour_index
				lowest_distance = distance_to_center
		(x,y),cell_radius = cv2.minEnclosingCircle(contours[lowest_index])        #obtem o ponto central e o raio da menor circunferencia possivel que engloba a celula mais proxima do centro da imagem (celula de interesse)
		cell_radius = int(cell_radius)                                            #normaliza o tipo de dado referente ao raio da "celula"
		cell_center = (int(x),int(y))                                             #normaliza o tipo de dado referente ao ponto central da celula
		contours.pop(lowest_index)                                                #remove o contorno da celula de interesse da lista que contem os contornos da imagem 
		return cell_center , cell_radius , contours




base = BaseLoader(train_base_path = 'bases/ALL' ,  validation_base_path = 'bases/Teste_ALL_IDB2/ALL')
base.load()

for image in base.train_images:
	rgb_image = cv2.imread(image.path)
	shifted = cv2.pyrMeanShiftFiltering(rgb_image, 10, 12)
	gray_image = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
	erythrocytes = get_erythrocytes(rgb_image)
	otsu = FloodBorders(gray_image , value = 0).process()
	erythrocytes_inverted = cv2.bitwise_not(erythrocytes)           #invertea imagem para as hemacias para que as hemacias fiquem em preto e o fundo branco
	ones_mask = cv2.threshold(erythrocytes_inverted,100,1,cv2.THRESH_BINARY)[1] 
	result = otsu * ones_mask
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))   #kernel circular usado para operacao de abertura     
	closing = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)  #aplica fechamento para remover ruidos (alguns contornos da celila principal podem ter ficado na imagem resultante da som)
	ones_mask = cv2.threshold(closing,10,1,cv2.THRESH_BINARY)[1]
	result = rgb_image * cv2.merge((ones_mask , ones_mask , ones_mask))
	gray_image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
	binary = cv2.threshold(gray_image,148,255,cv2.THRESH_BINARY)[1]
	mask = cv2.bitwise_not(binary)
	ones_mask = cv2.threshold(mask,10,1,cv2.THRESH_BINARY)[1]
	result = gray_image * ones_mask
	binary = cv2.threshold(result,10,255,cv2.THRESH_BINARY)[1]
	contours_image , contours, hierarchy = cv2.findContours(binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #metodo usado para recuperar os contornos de uma imagem binaria
	cv2.drawContours(contours_image, contours, -1,255, 1)  #desenha os contornos na imagem retornada pelo metodo cv2.findContours
	cell_center , cell_radius , contours = find_interest_cell(contours , rgb_image)
	result = RegionGrowing(contours_image , seed = cell_center ,  value = 255).process()
	opening = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
	ones_mask = cv2.threshold(closing,100,1,cv2.THRESH_BINARY)[1]
	result = rgb_image * cv2.merge((ones_mask , ones_mask , ones_mask))
	#cv2.imshow('rgb' , rgb_image)
	#cv2.imshow('shifted' , shifted)
	#cv2.imshow('otsu' , otsu)
	#cv2.imshow('erythrocytes' , erythrocytes_inverted)
	#cv2.imshow('result' , result)
	#cv2.imshow('closing' , closing)
	cv2.imshow('reseult' , result)
	#cv2.imshow('closing' , closing)
	#cv2.imshow('gray' , contours_image)
	cv2.waitKey(0)
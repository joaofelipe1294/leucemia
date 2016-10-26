import cv2
import math
import numpy as np
from modules.base.base_loader import BaseLoader
from modules.image_processing.image_chanels import ImageChanels
from modules.image_processing.filters import *


def process(image_path):
	img = cv2.imread(image_path)
	gray_image = cv2.imread(image_path , 0)
	gray_flooded = FloodBorders(gray_image).process()
 	binary_flooded = cv2.threshold(gray_flooded,127,255,cv2.THRESH_BINARY)[1]
 	h , s = ImageChanels(img).hsv()[:2]	
 	#binary_s = OtsuThresholdFilter().process(s)
 	binary_h = cv2.adaptiveThreshold(h , 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
 	binary_s = cv2.adaptiveThreshold(s,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	median_s = cv2.medianBlur(binary_s , 5)
	median_h = cv2.medianBlur(binary_h , 5)
	result = cv2.bitwise_and(median_s , median_h)
	erode = cv2.erode(result,np.ones((3,3) , np.uint8),iterations = 1)
	closing = cv2.morphologyEx(result, cv2.MORPH_OPEN, np.ones((3,3) , np.uint8))
	flooded_image = FloodBorders(closing).process()
	#opening = cv2.morphologyEx(flooded_image, cv2.MORPH_OPEN, np.ones((3,3) , np.uint8))
	and_image = cv2.bitwise_and(gray_flooded , closing)
	otsu = OtsuThreshold(and_image).process()

	"""
	contour_image, contours, hierarchy = cv2.findContours(flooded_image.copy() , cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cell_center , cell_radius , contours = find_interest_cell(contours , gray_image.shape[0] , gray_image.shape[1])

	seeds = []
	for ray in xrange(1,cell_radius / 5):
		seeds += houghs_transform(ray , cell_center)
	cleaned = clean_cell(closing , seeds)
	inverted = cv2.bitwise_not(cleaned)
	and_image = cv2.bitwise_and(closing , inverted)
	"""
	contour_image, contours, hierarchy = cv2.findContours(otsu.copy() , cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	for contour in contours:
		hull = cv2.convexHull(contour)
		contour_area = cv2.contourArea(contour)
	 	if contour_area > 300 and contour_area < 15000:
			cv2.drawContours(img,[hull],0,(0,0,255),2)  # draw contours in red color

	
 	cv2.imshow('original' , img)
	cv2.imshow('closing' , closing)
	cv2.imshow('gray_flooded' , gray_flooded)
	cv2.imshow('binary_flooded' , binary_flooded)
	cv2.imshow('yo' , otsu)
	#cv2.imshow('cleaned' , cleaned)
	#cv2.imshow('inverted' , inverted)
	#cv2.imshow('and' , and_image)
	cv2.waitKey(0)

	

def find_interest_cell(contours , height , width):
		#recebe os contornos de uma imagem e com base neles retorna o ponto central e o raio da celula de interesse , alem de uma lista com os contornos referente apenas a objetos que nao sejam a celula de interesse. A celula de interesse eh aquela que possui uma menor distancia Euclidiana de seu ponto central em relacao ao ponto central da imagem
		image_center_point = tuple([int(height / 2) , int(width) / 2])               #descobre o ponto central da imagem
		lowest_index = 0
		lowest_distance = None
		for contour_index in range(0 , len(contours)):                                         #itera sobre todos os contornos   
			(x,y) , object_radius = cv2.minEnclosingCircle(contours[contour_index])            #aplica a funcao cv2.minEnclosingCircle() que recebe um contorno e reorna o raio e o ponto central para a menor circunferencia  possivel que englobe o objeto referente ao contorno passado como parametro
			distance_to_center = math.sqrt(math.pow((x - image_center_point[1]) , 2) + math.pow(y - image_center_point[1] , 2))   #alcula a distancia Euclidiana para o ponto central do contorno em relacao ao ponto central da imagem , tendo em vista que a celula de interece eh a celula mais proxima ao centro da imagem 
			if lowest_distance == None or distance_to_center < lowest_distance:                #verifia se a distancia do ponto central do contorno e o ponto central da imagem eh menor do que a menor distancia computada anteriormente 
				lowest_index = contour_index
				lowest_distance = distance_to_center
		(x,y),cell_radius = cv2.minEnclosingCircle(contours[lowest_index])                     #obtem o ponto central e o raio da menor circunferencia possivel que engloba a celula mais proxima do centro da imagem (celula de interesse)
		cell_radius = int(cell_radius)                                                         #normaliza o tipo de dado referente ao raio da "celula"
		cell_center = (int(x),int(y))                                                          #normaliza o tipo de dado referente ao ponto central da celula
		contours.pop(lowest_index)                                                             #remove o contorno da celula de interesse da lista que contem os contornos da imagem 
		return cell_center , cell_radius , contours



def houghs_transform(ray , center): 
	seeds = []
	for teta in xrange(0,359 , 30):
		xt = int(center[0] - ray * math.cos(teta))
		yt = int(center[1] - ray * math.sin(teta))
		seeds.append(tuple([xt , yt]))
	return seeds



def clean_cell(image , seeds):
	copy = image.copy()
	for seed in seeds:
		copy = FloodFillFilter(copy).flood_region(seed)
	return copy


base = BaseLoader(train_base_path = 'bases/teste_segmentacao' ,  validation_base_path = 'bases/Teste_ALL_IDB2/ALL')
base.load()
#process(base.train_images[0].path)

for image in base.train_images:
	process(image.path)


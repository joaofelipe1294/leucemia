import cv2
import numpy as np
from modules.base.base_loader import BaseLoader
from modules.image_processing.image_chanels import ImageChanels
from modules.image_processing.filters import OtsuThreshold


def get_pxs(rgb_image , seed , kernel_size):
	'''metodo que pega os pixels centrais da imagem , ate o momento seta novos valores pra teste nos pixels alvo 
	kernel_size = 31
	seed = tuple([int(rgb_image.shape[0] / 2) , int(rgb_image.shape[1] / 2)])
	'''
	kernel_ray = int(kernel_size / 2)
	for x in xrange(seed[0] - kernel_ray, seed[0] + kernel_ray):
		for y in xrange(seed[1] - kernel_ray , seed[1] + kernel_ray):
			rgb_image.itemset((x , y , 0) , 255)
			rgb_image.itemset((x , y , 1) , 255)
			rgb_image.itemset((x , y , 2) , 255)

def set_pxs(gray_image , seed , kernel_size):
	'''metodo que pega os pixels centrais da imagem , ate o momento seta novos valores pra teste nos pixels alvo 
	kernel_size = 31
	seed = tuple([int(rgb_image.shape[0] / 2) , int(rgb_image.shape[1] / 2)])
	'''
	kernel_ray = int(kernel_size / 2)
	for x in xrange(seed[0] - kernel_ray, seed[0] + kernel_ray):
		for y in xrange(seed[1] - kernel_ray , seed[1] + kernel_ray):
			gray_image.itemset((x , y) , 1)
			gray_image.itemset((x , y) , 1)
			gray_image.itemset((x , y) , 1)



def get_erythocytes(rgb_image):
	blue_chanel = ImageChanels(rgb_image).rgb(chanel = 'B')         #separa canais RGB
	saturation = ImageChanels(rgb_image).hsv(chanel ='S')           #separa canais HSV
	saturation_binary = OtsuThreshold(saturation).process()         #aplica threshold na saturacao
	blue_chanel_binary = OtsuThreshold(blue_chanel).process()       #aplica threshold no canal B (azul)
	sum_image = blue_chanel_binary + saturation_binary              #soma os threshold da saturacao ao threshold do canal azul para remover a celula central da imagem , mantem apenas as hemacias em preto e o fundo branco
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))   #kernel circular usado para operacao de abertura     
	closing = cv2.morphologyEx(sum_image, cv2.MORPH_CLOSE, kernel)  #aplica fechamento para remover ruidos (alguns contornos da celila principal podem ter ficado na imagem resultante da som)
	result_image = cv2.bitwise_not(closing)                         #inverte a imgem resultante do fechamento , deixa apenas as hemacias em branco e o resto em preto
	return result_image


def apply_rgb_mask(rgb_image , mask_image):
		"""
			aplica uma mascara composta de zeros e uns propria para multiplicar duas imagens , trabalha com imagens RGB (3 canais) , retorna a imagem RGB
			apos a aplicacao da mascara nos tres canais
			parametros
				rgb_image  - imagem rgb (tres canais) que sera alterada pela mascara
				mask_image - imagem binaria , que tera seus valores alterados para zeros ou uns , depois sera montada uma matriz de trs dimensoes iguais , uma para cada um dos canais RGB
		"""
		ones_mask = cv2.threshold(mask_image,10,1,cv2.THRESH_BINARY)[1]   #aplica uma binarizacao que faz com que as hemacias fiquem com seus pixels com valor igual a 0 , isso eh usado porque a mascara eh resultado de uma multiplicacao
		mask = cv2.merge((ones_mask , ones_mask , ones_mask))              #monta uma matriz de tres dimensoes aonde cada uma das dimensoes eh a imagem binaria , formando um a mascara para cada um dos canais RGB
		result = rgb_image * mask                                          #multiplica as duas imagens , fazendo com que tudo que estaja com o valor 1 na mascara mantenha seu valor original e o que esta co o valor 0 na mascara suma
		return result



def mask_builder(image , kernel):
	kernel_ray = int(kernel / 2)                    #raio do kernel
	mask = np.zeros((image.shape[:2]) , np.uint8)   #criei mascara
	seeds = []     #lista que contem todas as coordenadas usadas como semente
	for x in xrange(kernel_ray , image.shape[1] , kernel * 2):   #vai de metade do kernel , ate o limite da largura da imagem , de 2 vezes o kernel por vez para criar um espaco esntre os blocos
		top_seed = tuple([kernel_ray, x])
		bottom_seed = tuple([image.shape[0] - kernel_ray , x])
		left_seed = tuple([x , kernel_ray])
		right_seed = tuple([x , image.shape[1] - kernel_ray])
		if x + kernel_ray < image.shape[1]:
			seeds.append(top_seed)
			seeds.append(bottom_seed)
			seeds.append(left_seed)
			seeds.append(right_seed)
		for seed in seeds:
			for x in xrange(seed[0] - kernel_ray, seed[0] + kernel_ray):
				for y in xrange(seed[1] - kernel_ray , seed[1] + kernel_ray):
					mask.itemset((x , y) , 1)
	return mask


def get_herythocytes_pxs(rgb_image):
	mask = mask_builder(rgb_image , 30)
	mask = cv2.merge((mask , mask , mask))
	erythrocytes = get_erythocytes(rgb_image)
	erythrocytes_rgb = apply_rgb_mask(rgb_image , erythrocytes)
	return erythrocytes_rgb * mask




rgb_image = cv2.imread('bases/ALL/Im001_1.tif')
erythrocytes = get_erythocytes(rgb_image)
inverted_erythrocytes = cv2.bitwise_not(erythrocytes)
erythrocytes_rgb = apply_rgb_mask(rgb_image , inverted_erythrocytes)
s = ImageChanels(erythrocytes_rgb).hsv('S')
otsu = OtsuThreshold(s).process()
otsu_not = cv2.bitwise_not(otsu)
free_center = apply_rgb_mask(erythrocytes_rgb , otsu_not)
mask = mask_builder(free_center , 30)
mask = cv2.merge((mask , mask , mask))
cv2.imshow('mask' , free_center * mask)
cv2.waitKey(0)


'''
base = BaseLoader(train_base_path = 'bases/ALL' ,  validation_base_path = 'bases/Teste_ALL_IDB2/ALL')
base.load()
for image in base.train_images:
	rgb_image = cv2.imread(image.path)
	erythrocytes = get_erythocytes(rgb_image)
	inverted_erythrocytes = cv2.bitwise_not(erythrocytes)
	erythrocytes_rgb = apply_rgb_mask(rgb_image , inverted_erythrocytes)
	s = ImageChanels(erythrocytes_rgb).hsv('S')
	otsu = OtsuThreshold(s).process()
	otsu_not = cv2.bitwise_not(otsu)
	free_center = apply_rgb_mask(erythrocytes_rgb , otsu_not)
	cv2.imshow('free_center' , free_center)
	cv2.imshow('original' , rgb_image)
	cv2.waitKey(0)
'''


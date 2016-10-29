import cv2
import numpy as np
from modules.base.base_loader import BaseLoader 
from modules.image_processing.image_chanels import ImageChanels
from modules.image_processing.filters import OtsuThreshold
from modules.image_processing.filters import FloodBorders
from modules.image_processing.contour import Contour
from modules.image_processing.filters import RegionGrowing
from modules.image_processing.segmentation import Segmentation



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
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))        #kernel 5, 5 redondo usado para aplicar morfologia matematica
	closing = cv2.morphologyEx(sum_image, cv2.MORPH_CLOSE, kernel)  #aplica fechamento para remover ruidos (alguns contornos da celila principal podem ter ficado na imagem resultante da som)
	result_image = cv2.bitwise_not(closing)                         #inverte a imgem resultante do fechamento , deixa apenas as hemacias em branco e o resto em preto
	return result_image


def get_interest_cell(image):
	"""
		extrai a celula central de uma imagem em tons de cinza , tem como objetivo retornar apenas a celula central em branco e o resto
		em preto , algumas vezes o resultado fica ruim , normalmente recebe como parametro a MATIZ ou a SATURACAO de uma imagem , 
		normalmente quando o resultado da MATIZ fica ruim o da SATURACAO eh satisfatorio , o contrario tambem eh valido.
		parametros
			@image - imagem em tons de ciza que sera processada , e tera sua celula central destacada
	"""
	binary_image = OtsuThreshold(image).process()                        	#binariza o canal da matiz , a binarizacao da saturacao tende a expor a celula como um todo , mas normalmente possui muitos ruidos 
	flooded_image = FloodBorders(binary_image , value = 0).process()        #inunda as bordas da matiz_binaria para remover os objetos presentes nas bordas
	contours , contours_image = Contour().get_contours(flooded_image)       #pega os contornos da matiz inundada
	filled_cell = RegionGrowing(contours_image , seed = (int(h.shape[0] / 2) , int(h.shape[1] / 2)) , value = 255).process() 	#aplica crescimento de recioes no ponto central da imagem , uma vez que a celula de interesse tende a ficar no centro da imagem , assim destadando a celula central das demais
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))                         #kernel circular usado para operacao de abertura     
	interest_cell = cv2.morphologyEx(filled_cell, cv2.MORPH_OPEN, kernel)            #aplica operacao de abertura para remover os contornos deixando apenas a celula central que estava inundada
	clean_interest_cell = cv2.threshold(interest_cell,127,255,cv2.THRESH_BINARY)[1]  #aplica um threshold com o objetivo de remover valores que nao sejam pequenos desaparecam
	return clean_interest_cell


def contour_area(binary_image):
	"""
		calcula a area total dos contornos de uma imagem binaria 
		parametros
			@binary_image - imagem binaria , essa imagem deve ser binaria para que os contornos sejam extraidos facilmente 
	"""
	contours , contours_image = Contour().get_contours(binary_image)     #pega os contronos da celula recebida como parametro
	if len(contours) == 1:                       #verifica se existe mais de um contorno
		return cv2.contourArea(contours[0])      #caso exista apenas um contorno retorna sua area
	else:                                        #senao itera sobre todos os contornos do array de contornos somando suas areas
		total_area = 0                          
		for contour in contours:                    
		 	total_area += cv2.contourArea(contour)
		return total_area

			
def apply_mask(rgb_image , mask_image):
	"""
		aplica uma mascara composta de zeros e uns propria para multiplicar duas imagens , trabalha com imagens RGB (3 canais) , retorna a imagem RGB
		apos a aplicacao da mascara nos tres canais
		parametros
			rgb_image  - imagem rgb (tres canais) que sera alterada pela mascara
			mask_image - imagem binaria , que tera seus valores alterados para zeros ou uns , depois sera montada uma matriz de trs dimensoes iguais , uma para cada um dos canais RGB
	"""
	ones_mask = cv2.threshold(mask_image,100,1,cv2.THRESH_BINARY)[1]   #aplica uma binarizacao que faz com que as hemacias fiquem com seus pixels com valor igual a 0 , isso eh usado porque a mascara eh resultado de uma multiplicacao
	mask = cv2.merge((ones_mask , ones_mask , ones_mask))              #monta uma matriz de tres dimensoes aonde cada uma das dimensoes eh a imagem binaria , formando um a mascara para cada um dos canais RGB
	result = rgb_image * mask                                          #multiplica as duas imagens , fazendo com que tudo que estaja com o valor 1 na mascara mantenha seu valor original e o que esta co o valor 0 na mascara suma
	return result



#base = BaseLoader(train_base_path = 'bases/teste_segmentacao' ,  validation_base_path = 'bases/Teste_ALL_IDB2/ALL')
base = BaseLoader(train_base_path = 'bases/teste_segmentacao' ,  validation_base_path = 'bases/Teste_ALL_IDB2/ALL')
base.load()
#process(base.train_images[0].path)

for image in base.train_images:
	print(image.path)
	rgb_image = cv2.imread(image.path)                              #le a imagem rgb
	erythrocytes = get_erythrocytes(rgb_image)                      #extrai as hemacias 
	erythrocytes_inverted = cv2.bitwise_not(erythrocytes)           #invertea imagem para as hemacias para que as hemacias fiquem em preto e o fundo branco
	erythrocytes_free = apply_mask(rgb_image , erythrocytes_inverted)
	h , s = ImageChanels(erythrocytes_free).hsv(display = False)[:2] #separa os canais da matiz e da saturacao da imagem livre de hemacias , agora passase a trabalhar com a matiz e a saturacao porque quando uma tem mal resultado o resultado do outro canal normalmente eh bom
	hue_cell = get_interest_cell(h)            #pega a celula central da matiz
	saturation_cell = get_interest_cell(s)     #pega a celula central da saturacao
	hue_area = contour_area(hue_cell)            #calcula a area da celula central presente na matiz
	saturation_area = contour_area(saturation_cell)     #calcula a area da celula central presente na saturacao
	mask = np.array((0,0))      #criada matriz vazia para ser usada como mascara
	if hue_area > saturation_area:    # caso a area dos contornos da MATIZ seja maior do que a area dos contornos da SATURACAO a matiz sera usada como mascara
		mask = hue_cell
	else:                             #caso contrario a SATURACAO eh usada como mascara
		mask = saturation_cell
	segmented_image = apply_mask(rgb_image , mask)   #aplica a mascara na imagem original , assim segmentando a celula central 

	
	cv2.imshow('alternativa' , segmented_image)
	cv2.imshow('primaria' , Segmentation(image.path).process())
	cv2.imshow('original' , rgb_image)
	cv2.waitKey(300)
import cv2
import numpy as np
from modules.base.base_loader import BaseLoader
from modules.image_processing.image_chanels import ImageChanels
from modules.image_processing.filters import *
from modules.image_processing.segmentation import Homogenization



"""
LABELS
	0 - center
	1 - hemacias
	2 - fundo
"""



"""
	s = ImageChanels(rgb_image).hsv('S' , display = False)
	erythrocytes = get_erythocytes(rgb_image)
	inverse_erythrocythes = cv2.bitwise_not(erythrocytes)
	erythrocytes_mask = cv2.threshold(inverse_erythrocythes,10,1,cv2.THRESH_BINARY)[1]
	otsu = OtsuThreshold(s).process()
	otsu = otsu * erythrocytes_mask
	mask = cv2.threshold(otsu,10,1,cv2.THRESH_BINARY)[1]
	mask = cv2.merge((mask , mask , mask))
	#
	h , s , v = ImageChanels(rgb_image).hsv(display = False)
	erythrocytes = get_erythocytes(rgb_image)
	mask_image = cv2.bitwise_not(erythrocytes)
	mask_image = cv2.threshold(mask_image,10,1,cv2.THRESH_BINARY)[1]
	mask = s * mask_image
	#cv2.imshow('mask' , mask)
	mask = cv2.threshold(mask,45,1,cv2.THRESH_BINARY)[1]
	mask = cv2.merge((mask , mask , mask))
	#cv2.imshow('get_pxs' , rgb_image * mask)
	#cv2.waitKey(0)
"""



#def get_pxs(rgb_image , kernel_size):
def get_pxs(rgb_image):
	shifted = cv2.pyrMeanShiftFiltering(rgb_image, 10, 12)
	hemacias = get_erythocytes(shifted)
	hemacias_inv = cv2.bitwise_not(hemacias)
	mask = cv2.threshold(hemacias_inv , 10 , 1 , cv2.THRESH_BINARY)[1]
	mask = cv2.merge((mask , mask , mask))
	gray = cv2.cvtColor(shifted * mask, cv2.COLOR_BGR2GRAY)
	gray = FloodBorders(gray).process()
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
	opening = cv2.morphologyEx(gray ,  cv2.MORPH_OPEN, kernel)
	#inicio pegar celula de interesse
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
	binary_image = cv2.threshold(opening , 10 , 255 , cv2.THRESH_BINARY)[1]
	flooded_image = FloodBorders(binary_image , value = 0).process()        #inunda as bordas da matiz_binaria para remover os objetos presentes nas bordas
	contours , contours_image = get_contours(flooded_image)       #pega os contornos da matiz inundada
	filled_cell = RegionGrowing(contours_image , seed = (int(rgb_image.shape[0] / 2) , int(rgb_image.shape[1] / 2)) , value = 255).process() 	#aplica crescimento de recioes no ponto central da imagem , uma vez que a celula de interesse tende a ficar no centro da imagem , assim destadando a celula central das demais
	interest_cell = cv2.morphologyEx(filled_cell, cv2.MORPH_OPEN, kernel)            #aplica operacao de abertura para remover os contornos deixando apenas a celula central que estava inundada
	clean_interest_cell = cv2.threshold(interest_cell,127,255,cv2.THRESH_BINARY)[1]  #aplica um threshold com o objetivo de remover valores que nao sejam pequenos desaparecam
	#fim pegar celula de interesse
	#cv2.imshow('interest_cell', clean_interest_cell)
	#cv2.imshow('sehfted', opening)
	#cv2.imshow('original' , rgb_image)
	#cv2.waitKey(0)
	mask = cv2.threshold(clean_interest_cell , 10 , 1 , cv2.THRESH_BINARY)[1]
	mask = cv2.merge((mask , mask , mask))
	return rgb_image * mask


def get_interest_cell(image):
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
	binary_image = OtsuThreshold(image).process()                        	#binariza o canal da matiz , a binarizacao da saturacao tende a expor a celula como um todo , mas normalmente possui muitos ruidos 
	flooded_image = FloodBorders(binary_image , value = 0).process()        #inunda as bordas da matiz_binaria para remover os objetos presentes nas bordas
	contours , contours_image = get_contours(flooded_image)       #pega os contornos da matiz inundada
	filled_cell = RegionGrowing(contours_image , seed = (int(image.shape[0] / 2) , int(image.shape[1] / 2)) , value = 255).process() 	#aplica crescimento de recioes no ponto central da imagem , uma vez que a celula de interesse tende a ficar no centro da imagem , assim destadando a celula central das demais
	interest_cell = cv2.morphologyEx(filled_cell, cv2.MORPH_OPEN, kernel)            #aplica operacao de abertura para remover os contornos deixando apenas a celula central que estava inundada
	clean_interest_cell = cv2.threshold(interest_cell,127,255,cv2.THRESH_BINARY)[1]  #aplica um threshold com o objetivo de remover valores que nao sejam pequenos desaparecam
	return clean_interest_cell

			
def get_contours(image):
	contours_image , contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #metodo usado para recuperar os contornos de uma imagem binaria
	cv2.drawContours(contours_image, contours, -1,255, 1)  #desenha os contornos na imagem retornada pelo metodo cv2.findContours
	return contours , contours_image



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


def get_herythocytes_pxs(rgb_image , kernel):
	mask = mask_builder(rgb_image , kernel)
	mask = cv2.merge((mask , mask , mask))
	erythrocytes = get_erythocytes(rgb_image)
	erythrocytes_rgb = apply_rgb_mask(rgb_image , erythrocytes)
	return erythrocytes_rgb * mask



def get_background_pxs(rgb_image , kernel):
	erythrocytes = get_erythocytes(rgb_image)
	inverted_erythrocytes = cv2.bitwise_not(erythrocytes)
	erythrocytes_rgb = apply_rgb_mask(rgb_image , inverted_erythrocytes)
	s = ImageChanels(erythrocytes_rgb).hsv('S')
	otsu = OtsuThreshold(s).process()
	otsu_not = cv2.bitwise_not(otsu)
	free_center = apply_rgb_mask(erythrocytes_rgb , otsu_not)
	mask = mask_builder(free_center , kernel)
	mask = cv2.merge((mask , mask , mask))
	return free_center * mask



def get_valid_values(rgb_image , label):
	valid_values = []
	hsv_center = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
	for x in xrange(0 , rgb_image.shape[0]):
		for y in xrange(0 , rgb_image.shape[1]):
			px_values = []
			px_values.append(rgb_image.item(x , y , 2))
			px_values.append(rgb_image.item(x , y , 1))
			px_values.append(rgb_image.item(x , y , 0))
			px_values.append(hsv_center.item(x , y , 0))
			px_values.append(hsv_center.item(x , y , 1))
			px_values.append(hsv_center.item(x , y , 2))			
			if all([ v != 0 for v in px_values ]):
				px_values.append(label) #adiciona a label na ultima posicao do array
				valid_values.append(px_values)
	return valid_values



"""
base = BaseLoader(train_base_path = 'bases/teste_segmentacao' ,  validation_base_path = 'bases/Teste_ALL_IDB2/ALL')
base.load()

base_values = []
for image in base.train_images:
	rgb_image = cv2.imread(image.path)
	center = get_pxs(rgb_image , 30)
	hemacias = get_herythocytes_pxs(rgb_image , 30)
	fundo = get_background_pxs(rgb_image , 30)
	center_values = get_valid_values(center , 0)
	erythrocytes_values = get_valid_values(hemacias , 1)
	background_values = get_valid_values(fundo , 1)    #alterado para que a label dos valores referentes ao fundo fique com a label 1
	#all_values = center_values + background_values
	all_values = center_values + erythrocytes_values + background_values
	base_values += all_values
	print(image.path)


file = open('valores_pxs.csv' , 'w')

for values in base_values:

	file.write(str(values[0]) + ',' + str(values[1]) + ',' + str(values[2]) + ',' + str(values[3]) + ',' + str(values[4]) + ',' + str(values[5]) + ',' + str(values[6]) + '\n')

file.close()
"""



'''
X = []
y = []
file = open('valores_pxs.csv' , 'r')
print('lendo arquivo csv ...')
for line in file:
	values = map(int , line.split(','))
	caracteristics = values[0: len(values) - 1]
	label = values[len(values) - 1]
	X.append(caracteristics)
	y.append(label)
file.close()
print('leitura concluida !')

from sklearn.svm import SVC
classifier = SVC(kernel="linear" , C = 0.025 , probability = True)
print('treinando classificador ...')
classifier.fit(X , y)
print('treinamento concluido !')


base = BaseLoader(train_base_path = 'bases/teste_segmentacao' ,  validation_base_path = 'bases/Teste_ALL_IDB2/ALL')
base.load()
for image in base.train_images:
	print('comecando segmentacao ...')
	rgb_image = cv2.imread(image.path)
	copy = rgb_image.copy()
	hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
	caracteristics = []
	for x in xrange(0 , rgb_image.shape[0]):
		for y in xrange(0 , rgb_image.shape[1]):
			r = rgb_image.item(x , y , 2)
			g = rgb_image.item(x , y , 1)
			b = rgb_image.item(x , y , 0)
			h = hsv_image.item(x , y , 0)
			s = hsv_image.item(x , y , 1)
			v = hsv_image.item(x , y , 2)
			caracteristics = [[r , g , b , h , s , v]]
			label = classifier.predict(caracteristics)
			if label[0] == 1:
				rgb_image.itemset((x , y , 0) , 127)
				rgb_image.itemset((x , y , 1) , 127)
				rgb_image.itemset((x , y , 2) , 127)
	print('segmentacao concluida !')
	cv2.imshow('yo' , rgb_image)
	cv2.imshow('original' , copy)
	cv2.waitKey(2000)	
'''




#########################################################
"""
base = BaseLoader(train_base_path = 'bases/teste_segmentacao' ,  validation_base_path = 'bases/Teste_ALL_IDB2/ALL')
base.load()
for image in base.train_images:
	rgb_image = cv2.imread(image.path)
	print('Extraindo valores dos pxs ...')
	center = get_pxs(rgb_image)
	hemacias = get_herythocytes_pxs(rgb_image , 30)
	fundo = get_background_pxs(rgb_image , 30)
	center_values = get_valid_values(center , 0)
	erythrocytes_values = get_valid_values(hemacias , 0)
	background_values = get_valid_values(fundo , 1)    #alterado para que a label dos valores referentes ao fundo fique com a label 1
	all_values = center_values + erythrocytes_values + background_values
	X = [values[0 : len(values) - 1] for values in all_values]
	y = [values[len(values) - 1] for values in all_values]
	#adicionado agora
	center_values_h = get_valid_values(center , 0)
	erythrocytes_values_h = get_valid_values(hemacias , 1)
	all_values_h = center_values_h + erythrocytes_values_h
	X_h = [values[0 : len(values) - 1] for values in all_values_h]
	y_h = [values[len(values) - 1] for values in all_values_h]


	print('Concluida extracao dos valores dos pxs !')
	from sklearn.svm import SVC
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
	from sklearn.ensemble import AdaBoostClassifier
	from sklearn.neighbors import KNeighborsClassifier
	classifier = SVC(kernel="linear" , C = 0.025 , probability = True)
	classifier_h = SVC(kernel="linear" , C = 0.025 , probability = True)
	#classifier = LinearDiscriminantAnalysis()
	#classifier =  AdaBoostClassifier()
	#classifier = KNeighborsClassifier(3)
	print('treinando classificador ...')
	classifier.fit(X , y)
	classifier_h.fit(X_h , y_h) #adicionado agora
	print('treinamento concluido !')
	print('Comecando a varrer a imagem ...')
	original = rgb_image.copy()
	hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
	for x in xrange(0 , rgb_image.shape[0]):
		for y in xrange(0 , rgb_image.shape[1]):
			r = rgb_image.item(x , y , 2)
			g = rgb_image.item(x , y , 1)
			b = rgb_image.item(x , y , 0)
			h = hsv_image.item(x , y , 0)
			s = hsv_image.item(x , y , 1)
			v = hsv_image.item(x , y , 2)
			caracteristics = [[r , g , b , h , s , v]]
			label = classifier.predict(caracteristics)
			if label[0] == 1:
				rgb_image.itemset((x , y , 0) , 0)
				rgb_image.itemset((x , y , 1) , 0)
				rgb_image.itemset((x , y , 2) , 0)
			else:
				label = classifier_h.predict(caracteristics)
				#print(label)
				if label[0] == 1:
					rgb_image.itemset((x , y , 0) , 0)
					rgb_image.itemset((x , y , 1) , 0)
					rgb_image.itemset((x , y , 2) , 0)
			#print('X :' + str(x) + ' | Y : ' + str(y))
	print('Segmentacao concluida !')
	from modules.image_processing.segmentation import ErythrocytesRemoval
	segmented_image = ErythrocytesRemoval(image.path).process()
	cv2.imshow('erythrocytes_removal' , segmented_image)
	cv2.imshow('px_px' , rgb_image)
	cv2.imshow('Original' , original)
	cv2.waitKey(0)
"""
#############################################################

'''
rgb_image = cv2.imread('bases/ALL/Im110_1.tif')
center_pxs = get_pxs(rgb_image)
#h , s , v = ImageChanels(rgb_image).hsv(display = False)
#erythrocytes = get_erythocytes(rgb_image)
#mask_image = cv2.bitwise_not(erythrocytes)
#mask_image = cv2.threshold(mask_image,10,1,cv2.THRESH_BINARY)[1]
#clear_image = s * mask_image
#clear_image = cv2.threshold(clear_image,60,255,cv2.THRESH_BINARY)[1]
cv2.imshow('original' , rgb_image)
cv2.imshow('hemacias' , center_pxs)
cv2.waitKey(0)
'''


#rgb_image = cv2.imread('bases/ALL/Im173_0.tif')
#get_pxs(rgb_image)



from modules.image_processing.filters import FloodBorders
count = 229
base = BaseLoader(train_base_path = 'bases/teste_segmentacao' ,  validation_base_path = 'bases/Teste_ALL_IDB2/ALL')
base.load()
for image in base.train_images:
	rgb_image = cv2.imread(image.path)
	print('Extraindo valores dos pxs ...')
	center = get_pxs(rgb_image)
	hemacias = get_herythocytes_pxs(rgb_image , 30)
	fundo = get_background_pxs(rgb_image , 30)
	center_values = get_valid_values(center , 0)
	erythrocytes_values = get_valid_values(hemacias , 0)
	background_values = get_valid_values(fundo , 1)    #alterado para que a label dos valores referentes ao fundo fique com a label 1
	all_values = center_values + erythrocytes_values + background_values
	X = [values[0 : len(values) - 1] for values in all_values]
	y = [values[len(values) - 1] for values in all_values]
	center_values_h = get_valid_values(center , 0)
	erythrocytes_values_h = get_valid_values(hemacias , 1)
	all_values_h = center_values_h + erythrocytes_values_h
	X_h = [values[0 : len(values) - 1] for values in all_values_h]
	y_h = [values[len(values) - 1] for values in all_values_h]
	print('Concluida extracao dos valores dos pxs !')
	from sklearn.svm import SVC
	classifier = SVC(kernel="linear" , C = 0.025 , probability = True)
	classifier_h = SVC(kernel="linear" , C = 0.025 , probability = True)
	print('treinando classificador ...')
	classifier.fit(X , y)
	classifier_h.fit(X_h , y_h) #adicionado agora
	print('treinamento concluido !')
	print('Comecando a varrer a imagem ...')
	original = rgb_image.copy()
	hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
	plano_fundo = np.zeros(rgb_image.shape[:2] , np.uint8)
	for x in xrange(0 , rgb_image.shape[0]):
		for y in xrange(0 , rgb_image.shape[1]):
			r = rgb_image.item(x , y , 2)
			g = rgb_image.item(x , y , 1)
			b = rgb_image.item(x , y , 0)
			h = hsv_image.item(x , y , 0)
			s = hsv_image.item(x , y , 1)
			v = hsv_image.item(x , y , 2)
			caracteristics = [[r , g , b , h , s , v]]
			label = classifier.predict(caracteristics)
			if label[0] == 1:
				plano_fundo.itemset(x , y , 255)
			else:
				label = classifier_h.predict(caracteristics)
				#print(label)
				if label[0] == 1:
					plano_fundo.itemset(x , y , 255)
				#rgb_image.itemset((x , y , 0) , 0)
				#rgb_image.itemset((x , y , 1) , 0)
				#rgb_image.itemset((x , y , 2) , 0)
			#print('X :' + str(x) + ' | Y : ' + str(y))
	print('Segmentacao concluida !')
	#from modules.image_processing.segmentation import ErythrocytesRemoval
	#segmented_image = ErythrocytesRemoval(image.path).process()
	#cv2.imshow('erythrocytes_removal' , segmented_image)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) 
	closing = cv2.morphologyEx(plano_fundo, cv2.MORPH_CLOSE, kernel)
	interest_cell = get_interest_cell(cv2.bitwise_not(closing))
	mask = cv2.threshold(interest_cell , 10 , 1 , cv2.THRESH_BINARY)[1]
	mask = cv2.merge((mask , mask , mask))
	segmented_image = rgb_image * mask
	#cv2.imshow('interest' , interest_cell)
	#cv2.imshow('closing' , closing)
	#cv2.imshow('px_px' , rgb_image)
	cv2.imwrite('bases/ALL_RESULTS/' + str(count) +  '.tiff' , segmented_image)
	count += 1
	print('================================================')
	#cv2.imshow('segmented_image' , segmented_image)
	#cv2.imshow('Original' , original)
	#cv2.waitKey(0)
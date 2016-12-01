import cv2
import numpy as np
from modules.base.base_loader import BaseLoader
from modules.image_processing.image_chanels import ImageChanels
from modules.image_processing.filters import *
from modules.image_processing.segmentation import Homogenization
from modules.image_processing.filters import FloodBorders
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


"""
LABELS
	0 - center
	1 - hemacias
	2 - fundo
"""


def pre_segmentation(rgb_image):
	"""
		faz uma pre segmentacao na imagem , tem como objetivo segmentar pelo menos uma parte do nucleo e do citoplasma da celula central
	"""
	homogenized_image = cv2.pyrMeanShiftFiltering(rgb_image, 10, 12) #deixa a imagem mais homogenia ,tornando mais facil diferenciar os conteudos da imagem (celula central , fundo e hemacias)
	red_cells = get_red_cells(homogenized_image)  #pega as hemacias da imagem
	red_cells_inverted = cv2.bitwise_not(red_cells)     #inverte as hemacias da imagem , para que as hemacias fiquem com valor 0 e o resto com valor 255
	red_cells_free_image = apply_rgb_mask(homogenized_image , red_cells_inverted)  #aplica a mascara na imagem original , removendo as hemacias
	gray = cv2.cvtColor(red_cells_free_image , cv2.COLOR_BGR2GRAY)  #converte a imagem que teve suas hemacias removidas para tons de cinza
	gray_flooded = FloodBorders(gray).process()     #inunda as bordas da imagem em tons de cinza, feito com o proposito de deixar o fundo com valor igual a 0
	interest_cell = get_interest_cell(gray_flooded , binarization_method = 'BINARY') #recupera apenas a celula central com valores 255 e os demais valores com o valor 0
	segmented_image = apply_rgb_mask(rgb_image , interest_cell)#aplica essa mascara que na teoria contem apenas a celula central (pelo menos boa parte dela) com a imagem original , removendo o fundo e as hemacias
	return segmented_image


def get_interest_cell(image , binarization_method = 'OTSU'):
	#binarization_method eh referente ao tipo de binarizacao que sera feita , o metodo default eh OTSU , o outro eh uma binarizacao binaria 
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
	if binarization_method == 'OTSU':
		binary_image = OtsuThreshold(image).process()                        	#binariza o canal da matiz , a binarizacao da saturacao tende a expor a celula como um todo , mas normalmente possui muitos ruidos 
	elif binarization_method == 'BINARY':
		binary_image = cv2.threshold(image , 10 , 255 , cv2.THRESH_BINARY)[1]
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



def get_red_cells(rgb_image):
	blue_chanel = ImageChanels(rgb_image).rgb(chanel = 'B')         #separa canais RGB
	saturation = ImageChanels(rgb_image).hsv(chanel ='S')           #separa canais HSV
	saturation_binary = OtsuThreshold(saturation).process()         #aplica threshold na saturacao
	blue_chanel_binary = OtsuThreshold(blue_chanel).process()       #aplica threshold no canal B (azul)
	sum_image = blue_chanel_binary + saturation_binary              #soma os threshold da saturacao ao threshold do canal azul para remover a celula central da imagem , mantem apenas as hemacias em preto e o fundo branco
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))     #kernel circular usado para operacao de abertura     
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
	"""
		metodo com o objetivo de gerar uma mascara composta por quadrados com tabanho X (kernel) , esses quadrados ficam nas bordas da imagem. Eh usado para pegar pxs do fundo e das hemacias
		parametros
			image  - imagem usada para pegar as dimensoes da mascara
			kernel - medida do lado de um dos quadrados que irao compor a mascara
	"""
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


def get_red_cells_pxs(rgb_image , kernel):
	"""
		metodo que tem como objetivo extrair alguns pxs referentes as hemacias presentes na imagem, retorna uma imagem que contem apenas parte das hemacias presentes na imagem original
		parametros
			rgb_image - imagem original 
			kernel    - medida de um dos lados do kernel que irao compor a ascara usada para extrair parte dos pixels da imagem
	"""
	red_cells = get_red_cells(rgb_image) #pega as hemacias da imagem
	red_cells_rgb = apply_rgb_mask(rgb_image , red_cells) #aplica mascara para que fiquem apenas as hemacias da imagem original em uma nova imagem RGB
	mask = mask_builder(rgb_image , kernel) #gera a mascara que ira preservar apenas os pixels das bordas das imagens
	mask = cv2.merge((mask , mask , mask)) #torna a mascara tridimensional
	segmented_image = red_cells_rgb * mask #multiplica a imagem RGB que contem apenas as hemacias pela mascara
	return segmented_image



def get_background_pxs(rgb_image , kernel):
	"""
		metodo que tem como objetivo extrair ps pxs do fundo da imagem , retorna uma imagem que contem apenas partes do fundo 
		parametros
			rgb_image - imagem original
			kernel    - medida de um dos lados do kernel que irao compor a ascara usada para extrair parte dos pixels da imagem
	"""
	red_cells = get_red_cells(rgb_image) #pega as hemacias da imagem
	inverted_red_cells = cv2.bitwise_not(red_cells) #inverte a imagem que contem apenas as hemacias para que ela fique com as hemacias em preto e o resto branco
	red_cells_free_rgb = apply_rgb_mask(rgb_image , inverted_red_cells) #usa a imagem que possui as hemacias com o valor 0 como mascara para remover as hemacias da imagem original RGB
	s = ImageChanels(red_cells_free_rgb).hsv('S') #pega o canal referente a saturacao
	otsu = OtsuThreshold(s).process()  #aplica um threshold de Otsu para que deixe apenas as celulas que possuem uma alta saturacao (celula central e celulas roxas mas que estao presentes nas bordas)
	otsu_not = cv2.bitwise_not(otsu) #inverte o resultado do otsu para que tudo que tenha uma alta saturacao fique preto e o resto fique branco
	free_center = apply_rgb_mask(red_cells_free_rgb , otsu_not) #usa o resultado da inversao do resultado do threshold como mascara com a finalidade de remover da imagem todos os pixels que possuem uma alta saturacao (celulas roxas)
	mask = mask_builder(free_center , kernel) #gera a mascara que tem como proposito extrair pixels do fundo
	mask = cv2.merge((mask , mask , mask)) #torna a mascara tridimensional
	segmented_image = free_center * mask #multiplica a imagem RGB resultante dos processos aplicados para remover as hemacias e as celulas roxas logo contendo apenas o fundo ou algo proximo a isso
	return segmented_image



def get_valid_values(rgb_image , label):
	"""
		metodo que retorna uma lista composta por listas que contem sete posicoes ,as seis primeiras posicoes sao referentes aos valores de cada um dos canais , e a ultima posicao eh a label desse tipo de pixel (CELULA CENTRAL , HEMACIA , FUNDO)
		parametros
			rgb_image - imagem que tera os valores dos seus pxs extraidos ,essa imagem eh apenas o fundo , apenas hemacias ou apennas celula central
			label     - label que sera atribuida aos vetores de caracteristicas
	"""
	valid_values = [] #cria a lista que ira conter os valores validos (pxs com pelo menos um dos canais diferente de 0)
	hsv_center = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV) #converte a imagem RGB para HSV
	for x in xrange(0 , rgb_image.shape[0]):
		for y in xrange(0 , rgb_image.shape[1]):
			px_values = [] #cria uma lista que ira conter os valores de cada um dos canais da imagem RGBHSV e a label que referese a hemacia , fundo e celula central
			px_values.append(rgb_image.item(x , y , 2))
			px_values.append(rgb_image.item(x , y , 1))
			px_values.append(rgb_image.item(x , y , 0))
			px_values.append(hsv_center.item(x , y , 0))
			px_values.append(hsv_center.item(x , y , 1))
			px_values.append(hsv_center.item(x , y , 2))			
			if all([ v != 0 for v in px_values ]):
				#px_values.append(label) #adiciona a label na ultima posicao do array
				valid_values.append(px_values)
	labels = [label] * len(valid_values)
	return valid_values , labels




from modules.image_processing.segmentation import SmartSegmentation
base = BaseLoader(train_base_path = 'bases/ALL' ,  validation_base_path = 'bases/Teste_ALL_IDB2/ALL')
base.load()
for image in base.train_images:
	SmartSegmentation(image.path).process(display = True)
"""	rgb_image = cv2.imread(image.path)
	print(image.path)
	print('Extraindo valores dos pxs ...')
	pre_segmented = pre_segmentation(rgb_image)  #pega imagem que contem apenas parte da celula central
	red_cells = get_red_cells_pxs(rgb_image , 30) #pega imagem que contem apenas parte da hemacias
	background = get_background_pxs(rgb_image , 30) #pega imagem que contem apenas parte do fundo
	pre_segmented_values , pre_segmented_labels = get_valid_values(pre_segmented , 0) #retorna uma lista com os valores de cada px valido da imagem que contem apenas a celula central ,atribuindo a label 0
	red_cells_values , red_cells_labels = get_valid_values(red_cells , 0) #retorna uma lista com os valores de cada px valido da imagem que contem apenas as hemacias ,atribuindo a label 0
	background_values , background_labels = get_valid_values(background , 1)  #alterado para que a label dos valores referentes ao fundo fique com a label 1
	X_background = pre_segmented_values + red_cells_values + background_values #concatena as caracteristicas de cada px formando um vetor de caracteristicas
	y_background = pre_segmented_labels + red_cells_labels + background_labels #concatena as listas compostas pelas label de cada vetor de caracteristica que forma X
	center_values_h , center_labels_h = get_valid_values(pre_segmented , 0) #atribui aos valores do centro a label 0
	red_cells_values_h , red_cells_labels_h = get_valid_values(red_cells , 1) #atribui as hemacias a label 1
	X_red_cells = center_values_h + red_cells_values_h #concatena as caracteristicas de cada px formando um vetor de caracteristicas
	y_red_cells = center_labels_h + red_cells_labels_h #concatena as listas compostas pelas label de cada vetor de caracteristica que forma X_red_cells
	print('Concluida extracao dos valores dos pxs !')
	classifier_background = LinearDiscriminantAnalysis() #cria classificador especialista em detectar o fundo
	classifier_red_cells = LinearDiscriminantAnalysis() #cria classificador especialista em detectar as hemacias
	print('treinando classificador ...')
	classifier_background.fit(X_background , y_background) #treian o classificador especialista em detectar fundo
	classifier_red_cells.fit(X_red_cells , y_red_cells) #treina o classificador especialista em detectar hemacias
	print('treinamento concluido !')
	print('Comecando a varrer a imagem ...')
	#original = rgb_image.copy()
	hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV) #cria uma imagem em HSV baseada na imagem original RGB
	segmentation_plan = np.zeros(rgb_image.shape[:2] , np.uint8) #cria imagem composta apenas por valores 0 e com as mesmas dimensoes da imagem que esta sendo segmentada
	for x in xrange(0 , rgb_image.shape[0]):      #varre a imagem px a px
		for y in xrange(0 , rgb_image.shape[1]):
			r = rgb_image.item(x , y , 2)  #pega os valores de cada um dos canais da imagem HSV e RGB para montar o vetor de caracteristicas
			g = rgb_image.item(x , y , 1)
			b = rgb_image.item(x , y , 0)
			h = hsv_image.item(x , y , 0)
			s = hsv_image.item(x , y , 1)
			v = hsv_image.item(x , y , 2)
			caracteristics = [[r , g , b , h , s , v]] #engloba os valores dos canais dentro de uma lista formando o vetor de caracteristicas
			label = classifier_background.predict(caracteristics) #verifica qual a label do px atual usando o classificador especialista em detectar pxs relativos ao fundo
			if label[0] == 1: #caso a label seja 1 (fundo) essa posicao no plano recebe o valor 255
				segmentation_plan.itemset(x , y , 255) #atribui o valor 255 a posicao x,y no plano que sera usado para segmentar a imagem
			else:
				label = classifier_red_cells.predict(caracteristics) #verifica a label do px atual segundo o classificador especialista em detectar pxs das hemacias
				if label[0] == 1: #caso a label seja 1 (hemacia) a posicao no plano de segmentacao recebe o valor 255 na posicao x,y
					segmentation_plan.itemset(x , y , 255) #atribui o valor 255 a posicao x,y no plano que sera usado para segmentar a imagem
	print('Segmentacao concluida !')
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))    
	closing = cv2.morphologyEx(segmentation_plan, cv2.MORPH_CLOSE, kernel) #aplica um fechamento no plano de segmentacao que no momento possui a celula de interesse em preto e o fundo e hemacias em branco para remover ruidos
	inverted_closing = cv2.bitwise_not(closing) #inverte a imagem resultate do fechamento para que a celula central fique branca e o resto em preto
	mask = get_interest_cell(inverted_closing) #recupera apenas a celula central da imagem , remove os demais elementos presentes na imagem 
	segmented_image = apply_rgb_mask(rgb_image , mask)
	cv2.imwrite('bases/ALL_RESULTS/LDA/' + str(count) +  '.tiff' , segmented_image)
	count += 1
	print('================================================')
	#cv2.imshow('segmented_image' , segmented_image)
	#cv2.imshow('Original' , original)
	#cv2.waitKey(0)
"""



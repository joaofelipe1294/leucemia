import cv2
import math
import numpy as np
from abc import ABCMeta , abstractmethod
from modules.base.base_loader import BaseLoader 
from modules.image_processing.image_chanels import ImageChanels
from modules.image_processing.filters import OtsuThreshold
from modules.image_processing.filters import FloodBorders
from modules.image_processing.filters import RegionGrowing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sys


##################################################################################################################


class Segmentation(object):
	"""
		classe abstrata usada para criar um padrao no desenvolvimento das classes que irao implementar os 
		pipelines de segmentacao , todas as classes com o objetivo de segmentar uma imagem devem herdar 
		de Segmentation e implementar o pipeline de segmentacao no metodo abstrato process()
	"""


	__metaclass__ = ABCMeta  #usado para tornas a classe filter abstrata

	
	def __init__(self , rgb_image):
		"""
			parametros
				@rgb_image - imagem que sera segmentada
			atributos
				@rgb_image  - imagem rgb lida a partir do caminho informado 
				@kernel     - elemento estrututante usado nas operacoes de morfologia matematica , elemento com formato de elipse 5x5
		"""
		self.rgb_image = rgb_image
		self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))   #kernel circular usado para operacao de abertura     


	@abstractmethod
	def process(self , display = False): 
		"""
			define o metodo abstrato que todas as classes filha devem implementar, esse metodo deve retornar a
		 	imagem resultante da aplicacao do pipiline que segmenta a imagem  
		"""
		pass	


	def get_red_cells(self , rgb_image):
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
		closing = cv2.morphologyEx(sum_image, cv2.MORPH_CLOSE, self.kernel)  #aplica fechamento para remover ruidos (alguns contornos da celila principal podem ter ficado na imagem resultante da som)
		result_image = cv2.bitwise_not(closing)                         #inverte a imgem resultante do fechamento , deixa apenas as hemacias em branco e o resto em preto
		return result_image


	def get_interest_cell(self , image , binarization_method = 'OTSU'):
		"""
			extrai a celula central de uma imagem em tons de cinza , tem como objetivo retornar apenas a celula central em branco e o resto
			em preto , algumas vezes o resultado fica ruim , normalmente recebe como parametro a MATIZ ou a SATURACAO de uma imagem , 
			normalmente quando o resultado da MATIZ fica ruim o da SATURACAO eh satisfatorio , o contrario tambem eh valido.
			parametros
				@image - imagem em tons de ciza que sera processada , e tera sua celula central destacada
				@binarization_method - define o algoritmo de binarizacao que sera utilisado ,no momento trabalha com OTSU ou BINARY
		"""
		binary_image = OtsuThreshold(image).process()                        	         #binariza o canal da matiz , a binarizacao da saturacao tende a expor a celula como um todo , mas normalmente possui muitos ruidos 
		if binarization_method == 'OTSU':
			binary_image = OtsuThreshold(image).process()                                #binariza o canal da matiz , a binarizacao da saturacao tende a expor a celula como um todo , mas normalmente possui muitos ruidos 
		elif binarization_method == 'BINARY':
			binary_image = cv2.threshold(image , 10 , 255 , cv2.THRESH_BINARY)[1]
		flooded_image = FloodBorders(binary_image , value = 0).process()                 #inunda as bordas da matiz_binaria para remover os objetos presentes nas bordas
		contours , contours_image = self.get_contours(flooded_image)                     #pega os contornos da matiz inundada
		filled_cell = RegionGrowing(contours_image , seed = (int(self.rgb_image.shape[0] / 2) , int(self.rgb_image.shape[1] / 2)) , value = 255).process() 	#aplica crescimento de recioes no ponto central da imagem , uma vez que a celula de interesse tende a ficar no centro da imagem , assim destadando a celula central das demais
		interest_cell = cv2.morphologyEx(filled_cell, cv2.MORPH_OPEN, self.kernel)       #aplica operacao de abertura para remover os contornos deixando apenas a celula central que estava inundada
		clean_interest_cell = cv2.threshold(interest_cell,127,255,cv2.THRESH_BINARY)[1]  #aplica um threshold com o objetivo de remover valores que nao sejam pequenos desaparecam
		return clean_interest_cell


	def apply_rgb_mask(self , rgb_image , mask_image):
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


	def get_contours(self , image):
		""""
			metodo usado para pegar os contornos de uma imagem , deve receber uma imagem binaria como parametro
			tem como objetivo retornar uma lista com as posicoes dos contornos (e o numero de contornos) e uma 
			imagem preta com os contornos desenhados em branco
			parametros 
				@image - imagem binaria que tera seus contornos extraidos
		"""
		contours_image , contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #metodo usado para recuperar os contornos de uma imagem binaria
		cv2.drawContours(contours_image, contours, -1,255, 1)  #desenha os contornos na imagem retornada pelo metodo cv2.findContours
		return contours , contours_image


	def find_interest_cell(self , contours):
		"""
			recebe os contornos de uma imagem e com base neles retorna o ponto central e o raio da celula de interesse
			, alem de uma lista com os contornos referente apenas a objetos que nao sejam a celula de interesse. A 
			celula de interesse eh aquela que possui uma menor distancia Euclidiana de seu ponto central em relacao 
			ao ponto central da imagem
			parametros
				@contours - lista com os contornos de uma imagem
		"""
		image_center_point = tuple([int(self.rgb_image.shape[0] / 2) , int(self.rgb_image.shape[1]) / 2])   #descobre o ponto central da imagem
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


	def display_image(self , image , time = 200):
		"""
			metodo usado para exibir uma imagem por um determinado tempo
			parametros
				@image - imagem que sera exibida
				@time  - tempo que a imagem sera exibida 
		"""
		cv2.imshow('segmented' , image)
		cv2.waitKey(time)


##################################################################################################################


##################################################################################################################


class ErythrocytesRemoval(Segmentation):
	"""
		proposta de segmentacao que primeiramente busca os pelas hemacias para gerar uma mascara , apos 
		multiplicar a mascara feita a partir das hemacias (o valor dos pixels das hemacias ficam com 
		valor 0 e os demais com valor 1) e a imagem original as hemacias sao completamente removidas
		.Depois passase a trabalhar com os canais de matiz (h) e saturacao (s), entao eh aplicado 
		threshold de Otsu na tentativa de isolar a celula central.Normalmente o resultado do threshold 
		da saturacao eh apenas o nucleo da celula , em alguns casos a celula eh perdida. Ja o resultado
		do threshold de Otsu na matiz na maioria das vezes eh a celula como um todo (nucleo e citoplasma) 
		o que eh considerado um resultado melhor do que apenas o nucleo da celula.Foi observado que quando
		a filtragem da saturacao tem um resultado ruim a filtragem da matiz tem um bom resultado ,o 
		contrario tambem eh valido.Apos segmentada a celula central eh extraido seu contorno e varificada 
		sua area ,para os resultados da matiz e saturacao, a imagem que apresentar uma maior area central
		eh a escolhida como resultado da segmentacao 
	"""


	def __init__(self , rgb_image):
		super(ErythrocytesRemoval , self).__init__(rgb_image = rgb_image)


	def get_contour_area(self , binary_image):
		"""
			calcula a area total dos contornos de uma imagem binaria 
			parametros
				@binary_image - imagem binaria , essa imagem deve ser binaria para que os contornos sejam extraidos facilmente 
		"""
		contours , contours_image = self.get_contours(binary_image)     #pega os contronos da celula recebida como parametro
		if len(contours) == 1:                       #verifica se existe mais de um contorno
			return cv2.contourArea(contours[0])      #caso exista apenas um contorno retorna sua area
		else:                                        #senao itera sobre todos os contornos do array de contornos somando suas areas
			total_area = 0                          
			for contour in contours:                    
			 	total_area += cv2.contourArea(contour)
			return total_area


	def process(self , display = False):
		erythrocytes = self.get_red_cells(self.rgb_image)                              #extrai as hemacias 
		erythrocytes_inverted = cv2.bitwise_not(erythrocytes)            #invertea imagem para as hemacias para que as hemacias fiquem em preto e o fundo branco
		erythrocytes_free = self.apply_rgb_mask(self.rgb_image , erythrocytes_inverted)
		h , s = ImageChanels(erythrocytes_free).hsv(display = False)[:2] #separa os canais da matiz e da saturacao da imagem livre de hemacias , agora passase a trabalhar com a matiz e a saturacao porque quando uma tem mal resultado o resultado do outro canal normalmente eh bom
		hue_cell = self.get_interest_cell(h)                             #pega a celula central da matiz
		saturation_cell = self.get_interest_cell(s)                      #pega a celula central da saturacao
		hue_area = self.get_contour_area(hue_cell)                           #calcula a area da celula central presente na matiz
		saturation_area = self.get_contour_area(saturation_cell)             #calcula a area da celula central presente na saturacao
		mask = np.array((0,0))                                           #criada matriz vazia para ser usada como mascara
		if hue_area > saturation_area:                                   # caso a area dos contornos da MATIZ seja maior do que a area dos contornos da SATURACAO a matiz sera usada como mascara
			mask = hue_cell
		else:                                                            #caso contrario a SATURACAO eh usada como mascara
			mask = saturation_cell
		segmented_image = self.apply_rgb_mask(self.rgb_image , mask)     #aplica a mascara na imagem original , assim segmentando a celula central 
		if display:
			self.display_image(segmented_image)
		return segmented_image


##################################################################################################################

##################################################################################################################

class FirstSegmentation(Segmentation):


	def __init__(self , rgb_image):
		self.rgb_image = cv2.imread(rgb_image)
		super(FirstSegmentation , self).__init__(rgb_image = rgb_image)


	def remove_noise_objects(self , contours_image , threshold_image , cell_center = None, cell_radius = None , contours = None):
		"""
			metodo com o objetivo de remover qualquer objeto da imagem que nao seja a celula de interesse
			parametros
				@contours_image  - imagem com apenas os contornos da celula desenhados
				@threshold_image - imagem binaria
				@cell_center     - ponto central da celula de interesse
				@cell_radius     - raio da celula de interesse 
		"""
		flooded_image = RegionGrowing(contours_image , seed = cell_center , value = 255).process()      #preenche a celula central 
		for contour in contours:                                                       #varre todos os contornos e verifica se a celula possui o nucleo vazado , nesse caso ocorre uma excecao que sera corrigida mais pra frente
			(x,y) , object_radius = cv2.minEnclosingCircle(contour)                    #computa o raio e o ponto central do contorno
			object_radius = int(object_radius)                                         #normaliza o tipo de dado  
			object_center = (int(x),int(y))                                            #normaliza o tipo de dado
			if (object_center[0] + object_radius > cell_center[0] + cell_radius and object_center[0] - object_radius < cell_center[0] - cell_radius) and (object_center[1] + object_radius > cell_center[1] + cell_radius and object_center[1] - object_radius < cell_center[1] + cell_radius): #verifica se o nucleo real da celula esta em volta do que foi marcado como nucleo
				opened_image = cv2.morphologyEx(threshold_image , cv2.MORPH_OPEN, self.kernel)
				return opened_image                                                                #nesse caso excepcional a imagem resultante do threshold de OTSU ja esta correta , eh aplicada uma operacao morfologica de abertura para remover pontos de ruido 
		opened_image = cv2.morphologyEx(flooded_image.copy(), cv2.MORPH_OPEN, self.kernel)   #remove os contornos de objetos que nao sejam a celula central
		return opened_image

	def process(self , display = False):                                      
		#faz a segmentacao da celula de interece   
		saturation = ImageChanels(self.rgb_image).hsv('S')                             #extraido canal relativo a Saturacao
		threshold_image = OtsuThreshold(saturation).process()					       #aplica threshold de OTSU no canal referente a saturacao 
		flooded_image = FloodBorders(threshold_image).process()                        #aplica o filtro flood_fill com o objetivo de remover os objetos colados as extremidades
		opened_image = cv2.morphologyEx(flooded_image, cv2.MORPH_OPEN, self.kernel)    #aplica operacao morfologica de abertura para remover pequenos pontos brancos (ruidos) presentes na imagem resultante da operacao anterior 
		contours , contour_image = self.get_contours(flooded_image)                    #computa uma imagem com os contornos desenhados e uma lista com aas coordenadas dos contornos 
		cell_center , cell_radius , contours = self.find_interest_cell(contours)       #computa o ponto central e o raio da celula de interesse 
		mask = None
		if len(contours) == 0:                                                         #se o numero de contornos for igual a zero significa que existe apenas um objeto na imagem opened_image logo a mascara ja esta correta
			mask = opened_image
		else:
			mask = self.remove_noise_objects(contour_image , threshold_image , cell_center = cell_center , cell_radius = cell_radius , contours = contours)
		segmented_image = self.apply_rgb_mask(self.rgb_image , mask)
		if display:
			self.display_image(segmented_image)
		return segmented_image


##################################################################################################################

##################################################################################################################


class Homogenization(Segmentation):
	"""
		o diferencial desse metodo de segmentacao eh que ele usa a homogenizacao da imagem RGB a partir da funcao 
		cv2.pyrMeanShiftFiltering() ,com a imagem mais "comportada" fica mais facil de extrair o fundo da imagem ,
		isso combinado ao metodo que extrai as hemacias apresentou um resultado satisfatorio ate certo ponto. 11%
		das imagens tiveram perdas significativas ,2% tiveram perda total e 87% tiveram bins resultados
	"""



	def __init__(self , rgb_image):
		self.rgb_image = cv2.imread(rgb_image)
		super(Homogenization , self).__init__(rgb_image = rgb_image)


	def apply_gray_scale_mask(self , gray_image , mask_image):
		"""
			aplica uma mascara composta por zeros e uns por meio da multiplicacao entre a imagem em tons de cinza e a mascara 
			parametros 
				gray_image - imagem em tons de cinza
				mask_image - imagem composta apenas por 0 e 1 usada como maskara
		"""
		ones_mask = cv2.threshold(mask_image,100,1,cv2.THRESH_BINARY)[1] #cria uma mascara com os valores 0 e 1		
		result = gray_image * ones_mask   #multiplica imagem em tons de cinza pela mascara ,tem como objetivo remover alguma parte especifica da imagem
		return result


	def process(self , display = False):
		shifted = cv2.pyrMeanShiftFiltering(self.rgb_image, 10, 12)                  #aplica o filtro MeanShiftFiltering , usado para homogenizar a imagem com o proposito de deixar o fundo com exatamente a mesma tonalidade
		gray_image = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)                       #converte a imagem para tons de cinza
		flooded_image = FloodBorders(gray_image , value = 0).process()               #inunda as bordas da imagem homogenizada que foi convertida para tons de cinza , isso remove o fundo e as hemacias que estao presentes nas bordas
		erythrocytes = self.get_red_cells(self.rgb_image)                            #obtem as hemacias da imagem
		erythrocytes_inverted = cv2.bitwise_not(erythrocytes)                        #invertea imagem para as hemacias para que as hemacias fiquem em preto e o fundo branco
		result = self.apply_gray_scale_mask(flooded_image , erythrocytes_inverted)   #multiplica as hemacias com valor 0 e a imagem inundada , com objetivo de remover a maior parte do fundo e das hemacias
		closing = cv2.morphologyEx(result, cv2.MORPH_OPEN, self.kernel)              #aplica abertura para remover pequenos ruidos da imagem resultante da multiplicacao
		result = self.apply_rgb_mask(self.rgb_image , closing)                       #usa a imagem resultante do fechamento como mascara e a multiplica pela imahem original
		gray_image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)                        #converte a imagem RGB (o que restou da remocao das hemacias e do fundo da imagem) para tons de cinza 
		background_removed = cv2.threshold(gray_image,148,255,cv2.THRESH_BINARY)[1]  #remove o fundo caso ele ainda continue na imagem mesmo apos a aplicacao dos passos anteriores
		mask = cv2.bitwise_not(background_removed)                                   #inverte o resultado do threshold deixando o fundo em preto e o resto da imagem em branco
		result = self.apply_gray_scale_mask(gray_image , mask)                       #tem como objetivo remover fundo das imagens em que ele ficou grudado a celula central
		binary = cv2.threshold(result,10,255,cv2.THRESH_BINARY)[1]                   #binariza a imagem para que seus contornos possam ser extraidos com uma maior precisao
		contours , contours_image = self.get_contours(binary)
		cell_center , cell_radius , contours = self.find_interest_cell(contours)              #pega a coordenada do ponto central da celula de interesse (celula que possui seu ponto central mais proximo do ponto central da imagem)
	 	result = RegionGrowing(contours_image , seed = cell_center ,  value = 255).process()  #aplica crescimento de regioes na celula central , usa o ponto central da celula central como semente
		opening = cv2.morphologyEx(result, cv2.MORPH_OPEN, self.kernel)                       #aplica abertura para remover os contornos dos demais objetos que nao a celula central
		closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.kernel)                     #aplica fechamento para remover os pontos pretos internos da celula central (binarizada)
		result = self.apply_rgb_mask(self.rgb_image , closing)
		if display:
			self.display_image(result)
		return result


##################################################################################################################

##################################################################################################################


class SmartSegmentation(Segmentation):
	"""
		metodo de segmentacao que utiliza o classificador LDA para classificar os pxs em fundo , hemacia ou celula
		de interesse. 
		1. Faz uma pre segmentacao buscando isolar cada uma das partes da celula
		2. Monta vetores de caracteristicas baseado nos valores de cada um dos canais RGB e HSV de cada px da imagem 
		3. Treina dois classificadores , um especialista em detectar o fundo (utilizado label 0 para hemacias e 
		celula central e 1 para o fundo) e outro especialista em detectar hemacias(utilizado label 0 para celula cen-
		tral e 1 para hemacias)
		4. Varre a imagem classificando cada px , e montando uma mascara que ira conter os pxs do fundo e das hemcais
		5 multiplica a mascara invertida pela imagem original, resultando na maioria dos casos em apenas a celula cen-
		tral e o resto preto
	"""


	def __init__(self , rgb_image):
		super(SmartSegmentation , self).__init__(rgb_image = rgb_image)


	def pre_segmentation(self):
		"""
			faz uma pre segmentacao na imagem , tem como objetivo segmentar pelo menos uma parte do nucleo e do 
			citoplasma da celula central. A ideia desse metodo eh primeiro deixar a imagem mais homogenea antes
			de segmentala, o que facilita na diferenciacao das partes que compoem uma imagem (celulas que reagi-
			ram com o reagente roxo , hemacias e fundo)
		"""
		homogenized_image = cv2.pyrMeanShiftFiltering(self.rgb_image, 10, 12)              #deixa a imagem mais homogenia ,tornando mais facil diferenciar os conteudos da imagem (celula central , fundo e hemacias)
		red_cells = self.get_red_cells(homogenized_image)                                  #pega as hemacias da imagem
		red_cells_inverted = cv2.bitwise_not(red_cells)                                    #inverte as hemacias da imagem , para que as hemacias fiquem com valor 0 e o resto com valor 255
		red_cells_free_image = self.apply_rgb_mask(homogenized_image , red_cells_inverted) #aplica a mascara na imagem original , removendo as hemacias
		gray = cv2.cvtColor(red_cells_free_image , cv2.COLOR_BGR2GRAY)                     #converte a imagem que teve suas hemacias removidas para tons de cinza
		gray_flooded = FloodBorders(gray).process()                                        #inunda as bordas da imagem em tons de cinza, feito com o proposito de deixar o fundo com valor igual a 0
		interest_cell = self.get_interest_cell(gray_flooded , binarization_method = 'BINARY')   #recupera apenas a celula central com valores 255 e os demais valores com o valor 0
		segmented_image = self.apply_rgb_mask(self.rgb_image , interest_cell)              #aplica essa mascara que na teoria contem apenas a celula central (pelo menos boa parte dela) com a imagem original , removendo o fundo e as hemacias
		return segmented_image


	def mask_builder(self , image , kernel = 30):
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


	def get_red_cells_pxs(self):
		"""
			metodo que tem como objetivo extrair alguns pxs referentes as hemacias presentes na imagem, retorna uma 
			imagem que contem apenas parte das hemacias presentes na imagem original
		"""
		red_cells = self.get_red_cells(self.rgb_image)                  #pega as hemacias da imagem
		red_cells_rgb = self.apply_rgb_mask(self.rgb_image , red_cells) #aplica mascara para que fiquem apenas as hemacias da imagem original em uma nova imagem RGB
		mask = self.mask_builder(self.rgb_image)                        #gera a mascara que ira preservar apenas os pixels das bordas das imagens
		mask = cv2.merge((mask , mask , mask))                          #torna a mascara tridimensional
		segmented_image = red_cells_rgb * mask                          #multiplica a imagem RGB que contem apenas as hemacias pela mascara
		return segmented_image


	def get_background_pxs(self):
		"""
			metodo que tem como objetivo extrair ps pxs do fundo da imagem , retorna uma imagem que contem apenas partes do fundo 
		"""
		red_cells = self.get_red_cells(self.rgb_image)                                #pega as hemacias da imagem
		inverted_red_cells = cv2.bitwise_not(red_cells)                               #inverte a imagem que contem apenas as hemacias para que ela fique com as hemacias em preto e o resto branco
		red_cells_free_rgb = self.apply_rgb_mask(self.rgb_image , inverted_red_cells) #usa a imagem que possui as hemacias com o valor 0 como mascara para remover as hemacias da imagem original RGB
		s = ImageChanels(red_cells_free_rgb).hsv('S')                                 #pega o canal referente a saturacao
		otsu = OtsuThreshold(s).process()                                             #aplica um threshold de Otsu para que deixe apenas as celulas que possuem uma alta saturacao (celula central e celulas roxas mas que estao presentes nas bordas)
		otsu_not = cv2.bitwise_not(otsu)                                              #inverte o resultado do otsu para que tudo que tenha uma alta saturacao fique preto e o resto fique branco
		free_center = self.apply_rgb_mask(red_cells_free_rgb , otsu_not)              #usa o resultado da inversao do resultado do threshold como mascara com a finalidade de remover da imagem todos os pixels que possuem uma alta saturacao (celulas roxas)
		mask = self.mask_builder(free_center)                                #gera a mascara que tem como proposito extrair pixels do fundo
		mask = cv2.merge((mask , mask , mask))                                        #torna a mascara tridimensional
		segmented_image = free_center * mask                                          #multiplica a imagem RGB resultante dos processos aplicados para remover as hemacias e as celulas roxas logo contendo apenas o fundo ou algo proximo a isso
		return segmented_image


	def get_valid_values(self , rgb_image , label):
		"""
			metodo que retorna uma lista composta por listas que contem sete posicoes ,as seis primeiras posicoes sao referentes aos valores de cada um dos canais , e a ultima posicao eh a label desse tipo de pixel (CELULA CENTRAL , HEMACIA , FUNDO)
			parametros
				@rgb_image - imagem grb que sera varrida e tera as os valores de cada px diferente de 0 extraidos e montados um vetor de caracteristicas
				@label     - label que sera atribuida aos vetores de caracteristicas
		"""
		valid_values = []                                            #cria a lista que ira conter os valores validos (pxs com pelo menos um dos canais diferente de 0)
		hsv_center = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV) #converte a imagem RGB para HSV
		for x in xrange(0 , rgb_image.shape[0]):
			for y in xrange(0 , rgb_image.shape[1]):
				px_values = []                                       #cria uma lista que ira conter os valores de cada um dos canais da imagem RGBHSV e a label que referese a hemacia , fundo e celula central
				px_values.append(rgb_image.item(x , y , 2))
				px_values.append(rgb_image.item(x , y , 1))
				px_values.append(rgb_image.item(x , y , 0))
				px_values.append(hsv_center.item(x , y , 0))
				px_values.append(hsv_center.item(x , y , 1))
				px_values.append(hsv_center.item(x , y , 2))			
				if all([ v != 0 for v in px_values ]):
					valid_values.append(px_values)
		labels = [label] * len(valid_values)
		return valid_values , labels


	def process(self , display = False):
		#print('Extraindo valores dos pxs ...')
		pre_segmented = self.pre_segmentation()                                                #pega imagem que contem apenas parte da celula central
		red_cells = self.get_red_cells_pxs()                                                   #pega imagem que contem apenas parte da hemacias
		background = self.get_background_pxs()                                                 #pega imagem que contem apenas parte do fundo
		pre_segmented_values , pre_segmented_labels = self.get_valid_values(pre_segmented , 0) #atribui a label 0 para os valores referentes a celula central
		red_cells_values , red_cells_labels = self.get_valid_values(red_cells , 0)             #atribui a label 0 para os valores referentes as hemacias 
		background_values , background_labels = self.get_valid_values(background , 1)          #atribui a label 1 para os valores referentes ao fundo da imagem
		X_background = pre_segmented_values + red_cells_values + background_values             #concatena as caracteristicas de cada px formando um vetor de caracteristicas
		y_background = pre_segmented_labels + red_cells_labels + background_labels             #concatena as listas compostas pelas label de cada vetor de caracteristica que forma X
		center_values_h , center_labels_h = self.get_valid_values(pre_segmented , 0)           #atribui a label 0 aos valores referentes a celula central
		red_cells_values_h , red_cells_labels_h = self.get_valid_values(red_cells , 1)         #atribui a label 1 aos valores referentes as hemacias
		X_red_cells = center_values_h + red_cells_values_h                    #concatena as caracteristicas de cada px formando um vetor de caracteristicas
		y_red_cells = center_labels_h + red_cells_labels_h                    #concatena as listas compostas pelas label de cada vetor de caracteristica que forma X_red_cells
		#print('Concluida extracao dos valores dos pxs !')
		classifier_background = LinearDiscriminantAnalysis()                  #cria classificador especialista em detectar o fundo
		classifier_red_cells = LinearDiscriminantAnalysis()                   #cria classificador especialista em detectar as hemacias
		#print('treinando classificador ...')
		classifier_background.fit(X_background , y_background)                #treian o classificador especialista em detectar fundo
		classifier_red_cells.fit(X_red_cells , y_red_cells)                   #treina o classificador especialista em detectar hemacias
		#print('treinamento concluido !')
		#print('Comecando a varrer a imagem ...')
		hsv_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)           #cria uma imagem em HSV baseada na imagem original RGB
		segmentation_plan = np.zeros(self.rgb_image.shape[:2] , np.uint8)     #cria imagem composta apenas por valores 0 e com as mesmas dimensoes da imagem que esta sendo segmentada
		for x in xrange(0 , self.rgb_image.shape[0]):                         #varre a imagem px a px
			for y in xrange(0 , self.rgb_image.shape[1]):
				r = self.rgb_image.item(x , y , 2)                            #pega os valores de cada um dos canais da imagem HSV e RGB para montar o vetor de caracteristicas
				g = self.rgb_image.item(x , y , 1)
				b = self.rgb_image.item(x , y , 0)
				h = hsv_image.item(x , y , 0)
				s = hsv_image.item(x , y , 1)
				v = hsv_image.item(x , y , 2)
				caracteristics = [[r , g , b , h , s , v]]                     #engloba os valores dos canais dentro de uma lista formando o vetor de caracteristicas
				label = classifier_background.predict(caracteristics)          #verifica qual a label do px atual usando o classificador especialista em detectar pxs relativos ao fundo
				if label[0] == 1:                                              #caso a label seja 1 (fundo) essa posicao no plano recebe o valor 255
					segmentation_plan.itemset(x , y , 255)                     #atribui o valor 255 a posicao x,y no plano que sera usado para segmentar a imagem
				else:
					label = classifier_red_cells.predict(caracteristics)       #verifica a label do px atual segundo o classificador especialista em detectar pxs das hemacias
					if label[0] == 1:                                          #caso a label seja 1 (hemacia) a posicao no plano de segmentacao recebe o valor 255 na posicao x,y
						segmentation_plan.itemset(x , y , 255)                 #atribui o valor 255 a posicao x,y no plano que sera usado para segmentar a imagem
		#print('Segmentacao concluida !')
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))    
		closing = cv2.morphologyEx(segmentation_plan, cv2.MORPH_CLOSE, kernel) #aplica um fechamento no plano de segmentacao que no momento possui a celula de interesse em preto e o fundo e hemacias em branco para remover ruidos
		inverted_closing = cv2.bitwise_not(closing)                            #inverte a imagem resultate do fechamento para que a celula central fique branca e o resto em preto
		mask = self.get_interest_cell(inverted_closing)                             #recupera apenas a celula central da imagem , remove os demais elementos presentes na imagem 
		segmented_image = self.apply_rgb_mask(self.rgb_image , mask)
		if display:
			self.display_image(segmented_image)
		return segmented_image


##################################################################################################################

##################################################################################################################

class SegmentNucleus(Segmentation):

	def __init__(self , rgb_image):
		super(SegmentNucleus , self).__init__(rgb_image = rgb_image)


	def process(self , display = False):
		s = ImageChanels(self.rgb_image).hsv(chanel = 'S' , display = False) #pega o canal referente a saturacao da imagem 
		for x in xrange(0,s.shape[0]):  #loop que deixa os pxs com valor 0 iguais a 127 para que o threshold de OTSU tenha um melhor resultado
			for y in xrange(0,s.shape[1]):
				if s.item(x,y) == 0:
					s.itemset(x,y,127)
		otsu = OtsuThreshold(s).process() #aplica o threshold de otsu na imagem referente a saturacao 'normalizada'
		erosion = cv2.erode(otsu , cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4)) ,iterations = 1)	#aplica morfologia do tipo erosao para fechar os contornos das celulas
		white = 0
		black = 0
		for x in xrange(0,otsu.shape[0]):  #verifica se a imagem ficou invertida
			for y in xrange(0,otsu.shape[1]):
				if otsu.item(x,y) == 255:
					white += 1
				else:
					black += 1
		if white > black:
			otsu = FloodBorders(erosion).process()
		segmented_image = FloodBorders(otsu).process() #inunda as bordas da imagem para remover celulas com alto gral de saturacao mas que nao sao a celula de interece
		if display:
			self.display_image(segmented_image)
		return segmented_image
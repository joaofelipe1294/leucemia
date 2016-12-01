import cv2
import math
import numpy as np
from abc import ABCMeta , abstractmethod
from modules.base.base_loader import BaseLoader 
from modules.image_processing.image_chanels import ImageChanels
from modules.image_processing.filters import OtsuThreshold
from modules.image_processing.filters import FloodBorders
from modules.image_processing.filters import RegionGrowing


##################################################################################################################


class Segmentation(object):
	"""
		classe abstrata usada para criar um padrao no desenvolvimento das classes que irao implementar os 
		pipelines de segmentacao , todas as classes com o objetivo de segmentar uma imagem devem herdar 
		de Segmentation e implementar o pipeline de segmentacao no metodo abstrato process()
	"""


	__metaclass__ = ABCMeta  #usado para tornas a classe filter abstrata

	
	def __init__(self , image_path = ''):
		"""
			parametros
				@image_path - string que contem o caminho da imagem que sera segmentada
			atributos
				@image_path - string que contem o caminho da imagem que sera segmentada
				@rgb_image  - imagem rgb lida a partir do caminho informado 
				@kernel     - elemento estrututante usado nas operacoes de morfologia matematica , elemento com formato de elipse 5x5
		"""
		self.image_path = image_path
		self.rgb_image = cv2.imread(self.image_path)
		self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))   #kernel circular usado para operacao de abertura     


	@abstractmethod
	def process(self , display = False): 
		"""
			define o metodo abstrato que todas as classes filha devem implementar, esse metodo deve retornar a
		 	imagem resultante da aplicacao do pipiline que segmenta a imagem  
		"""
		pass	


	def get_erythrocytes(self):
		"""
			extrai as hemacias de uma imagem RGB , o retorno eh uma imagem preta com apenas as hemacias em branco 
			parametros
				@rgb_image - imagem rgb que sera processada
		"""
		blue_chanel = ImageChanels(self.rgb_image).rgb(chanel = 'B')         #separa canais RGB
		saturation = ImageChanels(self.rgb_image).hsv(chanel ='S')           #separa canais HSV
		saturation_binary = OtsuThreshold(saturation).process()         #aplica threshold na saturacao
		blue_chanel_binary = OtsuThreshold(blue_chanel).process()       #aplica threshold no canal B (azul)
		sum_image = blue_chanel_binary + saturation_binary              #soma os threshold da saturacao ao threshold do canal azul para remover a celula central da imagem , mantem apenas as hemacias em preto e o fundo branco
		closing = cv2.morphologyEx(sum_image, cv2.MORPH_CLOSE, self.kernel)  #aplica fechamento para remover ruidos (alguns contornos da celila principal podem ter ficado na imagem resultante da som)
		result_image = cv2.bitwise_not(closing)                         #inverte a imgem resultante do fechamento , deixa apenas as hemacias em branco e o resto em preto
		return result_image


	def get_interest_cell(self , image):
		"""
			extrai a celula central de uma imagem em tons de cinza , tem como objetivo retornar apenas a celula central em branco e o resto
			em preto , algumas vezes o resultado fica ruim , normalmente recebe como parametro a MATIZ ou a SATURACAO de uma imagem , 
			normalmente quando o resultado da MATIZ fica ruim o da SATURACAO eh satisfatorio , o contrario tambem eh valido.
			parametros
				@image - imagem em tons de ciza que sera processada , e tera sua celula central destacada
		"""
		binary_image = OtsuThreshold(image).process()                        	#binariza o canal da matiz , a binarizacao da saturacao tende a expor a celula como um todo , mas normalmente possui muitos ruidos 
		flooded_image = FloodBorders(binary_image , value = 0).process()        #inunda as bordas da matiz_binaria para remover os objetos presentes nas bordas
		contours , contours_image = self.get_contours(flooded_image)       #pega os contornos da matiz inundada
		filled_cell = RegionGrowing(contours_image , seed = (int(self.rgb_image.shape[0] / 2) , int(self.rgb_image.shape[1] / 2)) , value = 255).process() 	#aplica crescimento de recioes no ponto central da imagem , uma vez que a celula de interesse tende a ficar no centro da imagem , assim destadando a celula central das demais
		interest_cell = cv2.morphologyEx(filled_cell, cv2.MORPH_OPEN, self.kernel)            #aplica operacao de abertura para remover os contornos deixando apenas a celula central que estava inundada
		clean_interest_cell = cv2.threshold(interest_cell,127,255,cv2.THRESH_BINARY)[1]  #aplica um threshold com o objetivo de remover valores que nao sejam pequenos desaparecam
		return clean_interest_cell


	def contour_area(self , binary_image):
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


	def __init__(self , image_path = ''):
		super(ErythrocytesRemoval , self).__init__(image_path = image_path)


	def process(self , display = False):
		erythrocytes = self.get_erythrocytes()                      #extrai as hemacias 
		erythrocytes_inverted = cv2.bitwise_not(erythrocytes)           #invertea imagem para as hemacias para que as hemacias fiquem em preto e o fundo branco
		erythrocytes_free = self.apply_rgb_mask(self.rgb_image , erythrocytes_inverted)
		h , s = ImageChanels(erythrocytes_free).hsv(display = False)[:2] #separa os canais da matiz e da saturacao da imagem livre de hemacias , agora passase a trabalhar com a matiz e a saturacao porque quando uma tem mal resultado o resultado do outro canal normalmente eh bom
		hue_cell = self.get_interest_cell(h)            #pega a celula central da matiz
		saturation_cell = self.get_interest_cell(s)     #pega a celula central da saturacao
		hue_area = self.contour_area(hue_cell)            #calcula a area da celula central presente na matiz
		saturation_area = self.contour_area(saturation_cell)     #calcula a area da celula central presente na saturacao
		mask = np.array((0,0))      #criada matriz vazia para ser usada como mascara
		if hue_area > saturation_area:    # caso a area dos contornos da MATIZ seja maior do que a area dos contornos da SATURACAO a matiz sera usada como mascara
			mask = hue_cell
		else:                             #caso contrario a SATURACAO eh usada como mascara
			mask = saturation_cell
		segmented_image = self.apply_rgb_mask(self.rgb_image , mask)   #aplica a mascara na imagem original , assim segmentando a celula central 
		if display:
			self.display_image(segmented_image)
		return segmented_image


##################################################################################################################

##################################################################################################################

class FirstSegmentation(Segmentation):


	def __init__(self , image_path):
		self.rgb_image = cv2.imread(image_path)
		super(FirstSegmentation , self).__init__(image_path = image_path)


	def process(self , display = False):                                      
		#faz a segmentacao da celula de interece   
		saturation = ImageChanels(self.rgb_image).hsv('S')                                         #extraido canal relativo a Saturacao
		threshold_image = OtsuThreshold(saturation).process()									   #aplica threshold de OTSU no canal referente a saturacao 
		flooded_image = FloodBorders(threshold_image).process()                           #aplica o filtro flood_fill com o objetivo de remover os objetos colados as extremidades
		opened_image = cv2.morphologyEx(flooded_image, cv2.MORPH_OPEN, self.kernel)  #aplica operacao morfologica de abertura para remover pequenos pontos brancos (ruidos) presentes na imagem resultante da operacao anterior 
		contours , contour_image = self.get_contours(flooded_image)                                           #computa uma imagem com os contornos desenhados e uma lista com aas coordenadas dos contornos 
		cell_center , cell_radius , contours = self.find_interest_cell(contours)                            #computa o ponto central e o raio da celula de interesse 
		mask = None
		if len(contours) == 0:                                                                     #se o numero de contornos for igual a zero significa que existe apenas um objeto na imagem opened_image logo a mascara ja esta correta
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



	def __init__(self , image_path):
		self.rgb_image = cv2.imread(image_path)
		super(Homogenization , self).__init__(image_path = image_path)


	def process(self , display = False):
		shifted = cv2.pyrMeanShiftFiltering(self.rgb_image, 10, 12) #aplica o filtro MeanShiftFiltering , usado para homogenizar a imagem com o proposito de deixar o fundo com exatamente a mesma tonalidade
		gray_image = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)      #converte a imagem para tons de cinza
		flooded_image = FloodBorders(gray_image , value = 0).process()       #inunda as bordas da imagem homogenizada que foi convertida para tons de cinza , isso remove o fundo e as hemacias que estao presentes nas bordas
		erythrocytes = self.get_erythrocytes()                  #obtem as hemacias da imagem
		erythrocytes_inverted = cv2.bitwise_not(erythrocytes)       #invertea imagem para as hemacias para que as hemacias fiquem em preto e o fundo branco
		result = self.apply_gray_scale_mask(flooded_image , erythrocytes_inverted)  #multiplica as hemacias com valor 0 e a imagem inundada , com objetivo de remover a maior parte do fundo e das hemacias
		closing = cv2.morphologyEx(result, cv2.MORPH_OPEN, self.kernel)      #aplica abertura para remover pequenos ruidos da imagem resultante da multiplicacao
		result = self.apply_rgb_mask(self.rgb_image , closing)               #usa a imagem resultante do fechamento como mascara e a multiplica pela imahem original
		gray_image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)                #converte a imagem RGB (o que restou da remocao das hemacias e do fundo da imagem) para tons de cinza 
		background_removed = cv2.threshold(gray_image,148,255,cv2.THRESH_BINARY)[1]   #remove o fundo caso ele ainda continue na imagem mesmo apos a aplicacao dos passos anteriores
		mask = cv2.bitwise_not(background_removed)                                    #inverte o resultado do threshold deixando o fundo em preto e o resto da imagem em branco
		result = self.apply_gray_scale_mask(gray_image , mask)               #tem como objetivo remover fundo das imagens em que ele ficou grudado a celula central
		binary = cv2.threshold(result,10,255,cv2.THRESH_BINARY)[1]           #binariza a imagem para que seus contornos possam ser extraidos com uma maior precisao
		contours , contours_image = self.get_contours(binary)
		cell_center , cell_radius , contours = self.find_interest_cell(contours)        #pega a coordenada do ponto central da celula de interesse (celula que possui seu ponto central mais proximo do ponto central da imagem)
	 	result = RegionGrowing(contours_image , seed = cell_center ,  value = 255).process()   #aplica crescimento de regioes na celula central , usa o ponto central da celula central como semente
		opening = cv2.morphologyEx(result, cv2.MORPH_OPEN, self.kernel)           #aplica abertura para remover os contornos dos demais objetos que nao a celula central
		closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.kernel)         #aplica fechamento para remover os pontos pretos internos da celula central (binarizada)
		result = self.apply_rgb_mask(self.rgb_image , closing)
		if display:
			cv2.imshow('result' , result)
			cv2.waitKey(0)
		return result
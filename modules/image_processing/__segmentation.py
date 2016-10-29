import cv2
import numpy as np
from abc import ABCMeta , abstractmethod
from modules.base.base_loader import BaseLoader 
from modules.image_processing.image_chanels import ImageChanels
from modules.image_processing.filters import OtsuThreshold
from modules.image_processing.filters import FloodBorders
from modules.image_processing.filters import RegionGrowing


##################################################################################################################


class __Segmentation(object):
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
	def process(self): 
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


	def apply_mask(self , rgb_image , mask_image):
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


##################################################################################################################


##################################################################################################################


class ErythrocytesRemoval(__Segmentation):
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


	def process(self):
		erythrocytes = self.get_erythrocytes()                      #extrai as hemacias 
		erythrocytes_inverted = cv2.bitwise_not(erythrocytes)           #invertea imagem para as hemacias para que as hemacias fiquem em preto e o fundo branco
		erythrocytes_free = self.apply_mask(self.rgb_image , erythrocytes_inverted)
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
		segmented_image = self.apply_mask(self.rgb_image , mask)   #aplica a mascara na imagem original , assim segmentando a celula central 
		return segmented_image


##################################################################################################################
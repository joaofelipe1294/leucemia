import os
import shutil
from random import randint
from modules.models.image import Image


class Kfold(object):

	def __init__(self , k = 2 , base_path = ''):
		""" 
		@parameters 
			k          - Obrigatorio numero de grupos em que a base sera subdividida
			base_path  - Obrigatorio caminho para a base que contem todas as imagens
		"""
		self.k = k
		self.base_path = base_path
		self.train_path = 'bases/train_temp'               # caminho da pasta temporaria que ira conter as imagens de treino
		self.validation_path = 'bases/validation_temp'     # caminho da pasta temporaria que ira conter as imagens de validacao
		self.all_images = []
		self.pipeline()


	def pipeline(self):
		#metodo que engloba todo o fluxo de execucao da logica de um kfold
		self.build_directories()
		self.all_images = self.load_base_images()
		self.replace_images()


	def build_directories(self):
		#metodo que cria os diretorios temporarios que irao armazenar as imagens de treino e validacao
		os.makedirs(self.train_path)
		os.makedirs(self.validation_path)


	def load_base_images(self):
		#print("Carregando imagens de " + base_path + " ...")
		paths = os.listdir(self.base_path)
		paths.sort()
		images = []
		for path in paths:
			image_id = path[2:5]
			image_path = self.base_path + '/' + path
			label = path[len(path) - 5:len(path) - 4]
			image = Image(image_id = image_id , path = image_path , label = label)
			images.append(image)
		#print("Imagens carregadas")
		return images


	def replace_images(self):
		images_per_fold = (len(self.all_images) / self.k) / 2   # calcula o numero de imagens de cada classe que compoe um fold
		positive_images = []
		negative_images = []
		for image in self.all_images:
			if image.label  == 1:
				positive_images.append(image)
			else:
				negative_images.append(image)
		#validation_positives = []
		#validation_negatives = []
		validation_images = []
		for x in xrange(0 , images_per_fold):
			random_index = randint(0 , len(positive_images) - 1)
			validation_images.append(positive_images.pop(random_index))
			validation_images.append(negative_images.pop(random_index))
		for image in validation_images:
			shutil.copy(image.path , self.validation_path + '/' + image.path[11:])
		train_images = positive_images + negative_images
		for image in train_images:
			shutil.copy(image.path , self.train_path + '/' + image.path[11:])
		for image in validation_images:
			print(image.path[11:])
		print('Reparticao concluida !')


	def remove_temp_dirs(self):
		shutil.rmtree(self.train_path)
		shutil.rmtree(self.validation_path)


class KFold(object):
	#classe que automatiza a criacao de uma distribuicao de amostras do tipo kfold

	def __init__(self , base_path , k = 2):
		self.k = k
		self.base_path = base_path


	def pipeline(self):
		all_images = self.load_base()
		positives , negatives = self.divide_positives_and_negatives(all_images)
		kfolds = self.devide_in_folds(positives , negatives)
		try:
			self.remove_directories()	
			self.build_directories()
		except Exception as e:
			self.build_directories()
		self.move_image_to_fold(kfolds)
		


	def load_base(self):
		#metodo que le todas as imagens da base e retorna uma lista com essas imagens ja carregadas
		print("Carregando imagens de " + self.base_path + " ...")
		paths = os.listdir(self.base_path)
		paths.sort()
		images = []
		for path in paths:
			image_id = path[2:5]
			image_path = self.base_path + '/' + path
			label = path[6] #5,4
			image = Image(image_id = image_id , path = image_path , label = label)
			images.append(image)
		print("Imagens carregadas")
		return images


	def divide_positives_and_negatives(self , images):
		#metodo que divide uma lista de Image em duas, aonde o criterio de divisao eh a label da imagem
		positives = []
		negatives = []
		for image in images:
			if image.label == 1:
				positives.append(image)
			else:
				negatives.append(image)
		return positives , negatives


	def devide_in_folds(self , positive_images , negative_images):
		#metodo que distribui as imagens em listas relativas aos folds
		kfolds = []
		number_of_images_per_fold = ((len(positive_images) + len(negative_images)) / self.k) / 2
		start_index = 0
		stop_index = number_of_images_per_fold
		for cont in xrange(0,self.k):
			fold_images = []
			for image_index in xrange(start_index,stop_index):
				fold_images.append(positive_images[image_index])
				fold_images.append(negative_images[image_index])
			kfolds.append(fold_images)
			start_index += number_of_images_per_fold
			stop_index += number_of_images_per_fold
		return kfolds


	def build_directories(self):
		#metodo que cria os diretorios temporarios que irao armazenar as imagens de treino e validacao
		os.makedirs('bases/kfolds')
		for x in xrange(1 , self.k + 1):
			os.makedirs('bases/kfolds/' + str(x) + 'fold')

	def remove_directories(self):
		shutil.rmtree('bases/kfolds')

	def move_image_to_fold(self , kfolds):
		k_index = 0
		print(kfolds)
		for images in kfolds:
			k_index += 1
			for image in images:
				#print(image.path[10:])
				shutil.copy(image.path , 'bases/kfolds/' + str(k_index) + 'fold/' + image.path[11:])
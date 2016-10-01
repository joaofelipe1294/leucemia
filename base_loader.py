# -*- coding: utf-8 -*-
import os
import sys
from image import Image
from segmentation import Segmentation
from feature_extractor import FeatureExtractor


class BaseLoader(object):


	def __init__(self , train_base_path = None , valid_base_path = None):
		self.train_base_path = train_base_path
		self.train_images = []
		self.train_file = "train_data.txt"
		self.train_vectors = []
		self.train_labels = []
		self.valid_base_path = valid_base_path
		self.valid_images = []
		self.valid_vectors = []
		self.valid_labels = []


	def load(self):
		if self.train_base_path:
			self.train_images = self.load_base_images(self.train_base_path)
			self.train_labels = self.get_labels(self.train_images)
			self.extract_features(file_path = self.train_file , images = self.train_images , train = True)
		else:
			self.load_train_vectors_from_file()
		self.valid_images = self.load_base_images(self.valid_base_path)
		self.valid_labels = self.get_labels(self.valid_images)
		self.extract_features(images = self.valid_images)


	def load_train_vectors_from_file(self):
		print("Carregando vetores ...")
		file = open( self.train_file, 'r')
		for line in file:
			values = line.split(',')
			area = int(values[0])
			variance = int(values[1])
			perimeter = int(values[2])
			excess = int(values[3])
			label = int(values[4])
			self.train_vectors.append([area , variance , perimeter , excess])
			self.train_labels.append(label)
		file.close()
		print("Vetores carregados")


	def load_base_images(self , base_path):
		print("Carregando imagens de " + base_path + " ...")
		paths = os.listdir(base_path)
		paths.sort()
		images = []
		for path in paths:
			image_id = path[2:5]
			image_path = base_path + '/' + path
			label = path[len(path) - 5:len(path) - 4]
			image = Image(image_id = image_id , path = image_path , label = label)
			images.append(image)
		print("Imagens carregadas")
		return images


	def get_labels(self , images):
		labels = []
		for image in images:
			labels.append(image.label)
		return labels


	def extract_features(self , file_path = None , images = None , train = False):
		base_features = []
		iteration = 0
		for image in images:
			segmented_image = Segmentation(image.path).process()
			area , variance , perimeter , excess = FeatureExtractor(segmented_image).get_features()
			base_features.append([area , perimeter , excess])
			iteration += 1
			self.printProgress(iteration , len(images) , prefix = "Treinamento : ")
		if train:
			file = open(file_path , 'w')
			iteration = 0
			for features in base_features:
				file.write(str(features[0]) + ' , ' + str(features[1]) + ' , ' + str(images[iteration].label) + '\n')
				iteration += 1
			file.close()
			self.train_vectors = base_features
		else:
			self.valid_vectors = base_features



	def printProgress (self , iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
	    """
	    Call in a loop to create terminal progress bar
	    @params:
	        iteration   - Required  : current iteration (Int)
	        total       - Required  : total iterations (Int)
	        prefix      - Optional  : prefix string (Str)
	        suffix      - Optional  : suffix string (Str)
	        decimals    - Optional  : positive number of decimals in percent complete (Int)
	        barLength   - Optional  : character length of bar (Int)
	    """
	    formatStr       = "{0:." + str(decimals) + "f}"
	    percents        = formatStr.format(100 * (iteration / float(total)))
	    filledLength    = int(round(barLength * iteration / float(total)))
	    bar             = '█' * filledLength + '-' * (barLength - filledLength)
	    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
	    if iteration == total:
	        sys.stdout.write('\n')
	    sys.stdout.flush()
# -*- coding: utf-8 -*-
import os
import sys
from image import Image
from segmentation import Segmentation
from feature_extractor import FeatureExtractor


class BaseLoader(object):


	def __init__(self , train_base_path = None):
		self.train_base_path = train_base_path
		self.train_images = []
		self.train_file = "train_data.txt"
		self.train_vectors = []


	def load(self):
		self.train_images = self.load_base_images(self.train_base_path)
		self.train(file_path = self.train_file , images = self.train_images , train = True)


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


	def train(self , file_path = None , images = None , train = False):
		file = open(file_path , 'w')
		iteration = 0
		for image in images:
			iteration += 1
			segmented_image = Segmentation(image.path).process()
			area , variance , perimeter , excess = FeatureExtractor(segmented_image).get_features()
			if train:
				file.write(str(area) + ' , ' + str(variance) + ' , ' + str(perimeter) + ' , ' + str(excess) + ' , ' + image.label + '\n')
				self.train_vectors.append([area , variance , perimeter , excess , image.label])
				self.printProgress(iteration , len(images) , prefix = "Treinamento : ")
			else:
				file.write(str(area) + ' , ' + str(variance) + ' , ' + str(perimeter) + ' , ' + str(excess) + '\n')
				self.printProgress(iteration , len(images) , prefix = "Validacao : ")
		file.close()



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
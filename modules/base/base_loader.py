# -*- coding: utf-8 -*-
import os
import sys
from modules.utils.progress_bar import ProgressBar
from modules.models.image import Image
from modules.image_processing.segmentation import Segmentation
from modules.features.feature_extractor import FeatureExtractor


class BaseLoader(object):


	def __init__(self , train_base_path = None , validation_base_path = None):
		self.train_base_path = train_base_path
		self.train_images = []
		self.train_labels = []
		self.validation_base_path = validation_base_path
		self.validation_images = []
		self.validation_labels = []


	def load(self):
		self.train_images = self.load_base_images(self.train_base_path)
		self.validation_images = self.load_base_images(self.validation_base_path)
		self.train_labels = self.get_labels(self.train_images)
		self.validation_labels = self.get_labels(self.validation_images)


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


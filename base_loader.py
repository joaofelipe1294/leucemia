import os
from image import Image

class BaseLoader(object):

	def __init__(self):
		self.base_path = None
		self.images = []

	def load(self ,base_path):
		self.base_path = base_path
		paths = os.listdir(base_path)
		paths.sort()
		for path in paths:
			image_id = path[2:5]
			image_path = self.base_path + '/' + path
			label = path[len(path) - 5:len(path) - 4]
			image = Image(image_id = image_id , path = image_path , label = label)
			self.images.append(image)

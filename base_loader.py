import os
from cell import Cell

class BaseLoader(object):

	def __init__(self):
		self.base_path = None
		self.cells = []

	def load(self ,base_path):
		self.base_path = base_path
		paths = os.listdir(base_path)
		paths.sort()
		for path in paths:
			image_id = path[2:5]
			image_path = self.base_path + '/' + path
			label = paths[0][6:7]
			cell = Cell(image_id = image_id , image_path = image_path , label = label)
			self.cells.append(cell)

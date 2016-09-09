class Cell(object):

	def __init__(self , image_id=None , image_path=None , label=None):
		self.image_id = image_id
		self.image_path = image_path
		self.label = label

	def __str__(self):
		return "id : " + str(self.image_id) + " | image_path : " + self.image_path + " | label : " + str(self.label)

class Image(object):

	def __init__(self , image_id=None , path=None , label=None):
		self.image_id = image_id
		self.path = path
		self.label = int(label)

	def __str__(self):
		return "id : " + str(self.image_id) + " | image_path : " + self.path + " | label : " + str(self.label)

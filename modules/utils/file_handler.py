class FileHandler:

	def __init__(self)
		self.train_file = 'train_data.txt'
		self.validation_file = 'validation_data.txt'

	def get_vetctors_and_labels(self , train = False , validation = False):
		file_path = ''
		if train:
			file_path = self.train_file
		elif validation:
			file_path = self.validation_file
		file = open(file_path , 'r')
		
		for line in file:
			if train:
				#continuar a carregar dados dos arquivos
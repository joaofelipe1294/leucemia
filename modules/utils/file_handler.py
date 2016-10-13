class FileHandler:

	def __init__(self):
		self.train_file = 'train_data.txt'
		self.validation_file = 'validation_data.txt'


	def load_vectors_and_labels(self , train = False , validation = False):
		vectors = []
		labels = []
		file_path = ''
		if train:
			file_path = self.train_file
		elif validation:
			file_path = self.validation_file
		file = open(file_path , 'r')
		for line in file:
			values = map(int, line.split(','))
			label = values.pop(len(values) - 1)
			vectors.append(values)
			labels.append(label)
		return vectors , labels
				
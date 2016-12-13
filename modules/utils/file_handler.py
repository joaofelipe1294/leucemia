class FileHandler:

	def __init__(self):
		self.train_file = 'train.csv'
		self.validation_file = 'validation.csv'


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
			values = map(float, line.split(','))
			label = values.pop(len(values) - 1)
			vectors.append(values)
			labels.append(label)
		return vectors , labels
				
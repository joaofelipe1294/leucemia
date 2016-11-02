from modules.utils.progress_bar import ProgressBar
from modules.image_processing.segmentation import FirstSegmentation
from modules.image_processing.segmentation import ErythrocytesRemoval
from modules.features.feature_extractor import FeatureExtractor


class BaseProcessor:

	def __init__(self):
		self.train_file = 'caracteristics_files/train_data.txt'
		self.validation_file = 'caracteristics_files/validation_data.txt'


	def load(self , images = None , train = False , validation = False):
		base_features = []
		iteration = 0
		for image in images:
			segmented_image = FirstSegmentation(image.path).process()
			area , perimeter , excess , average = FeatureExtractor(segmented_image).get_features()
			base_features.append([area , perimeter , excess])
			iteration += 1
			ProgressBar().printProgress(iteration , len(images) , prefix = "Treinamento : ")
		file_path = ''
		if train:
			file_path = self.train_file
		elif validation:
			file_path = self.validation_file

		file = open(file_path , 'w')
		iteration = 0
		for features in base_features:
			file.write(str(features[0]) +  ' , ' + str(features[1]) + ' , ' + str(features[2]) + ' , ' + str(images[iteration].label) + '\n')
			iteration += 1
		file.close()
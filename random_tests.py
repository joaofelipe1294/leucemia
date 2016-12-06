import cv2
import numpy as np
from modules.base.base_loader import BaseLoader
from modules.image_processing.segmentation import SmartSegmentation

base = BaseLoader(train_base_path = 'bases/ALL' ,  validation_base_path = 'bases/Teste_ALL_IDB2/ALL')
base.load()
for image in base.train_images:
	SmartSegmentation(image.path).process(display = True)
import cv2
import math
import numpy as np
from modules.base.base_loader import BaseLoader
from modules.image_processing.image_chanels import ImageChanels
from modules.image_processing.filters import *
from modules.image_processing.segmentation import Homogenization




base = BaseLoader(train_base_path = 'bases/teste_segmentacao' ,  validation_base_path = 'bases/Teste_ALL_IDB2/ALL')
base.load()

for image in base.train_images:
	print(image.path)                                      #imprime o caminho da imagem processada
	Homogenization(image.path).process(display = True)
	
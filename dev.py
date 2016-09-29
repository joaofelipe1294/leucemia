import cv2
import numpy as np
from base_loader import BaseLoader
from histogram import Histogram
from image_chanels import ImageChanels
from segmentation import Segmentation


base = BaseLoader()
#base.load('teste')
base.load('ALL_IDB2/img')
#base.load('Teste_ALL_IDB2/V0')
#base.load('Teste_ALL_IDB2/V1')
#base.load('teste_validacao')

for image in base.images:
	print("==============================================")
	print(image.path)
	segmented_image = Segmentation(image.path).process()
	#cv2.imshow('gray_image' , cv2.imread(image.path))
	#cv2.imshow('resault' , segmented_image)
	#cv2.waitKey(100)
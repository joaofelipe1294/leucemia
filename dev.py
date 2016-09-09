from base_loader import BaseLoader
import cv2
import numpy as np



base_loader = BaseLoader()
base_loader.load('ALL_IDB2/img')
image_path = base_loader.cells[0].image_path
rgb_image = cv2.imread(image_path)
gray_scale_image = cv2.imread(image_path , 0)
cv2.imshow('rgb' , rgb_image)
cv2.imshow('gray' , gray_scale_image)
cv2.waitKey(0)
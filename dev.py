from base_loader import BaseLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from segmentation import *
from image_chanels import *
import numpy as np
import cv2
from feature_extractor import *



#base = BaseLoader(valid_base_path = 'validacao')
#base = BaseLoader(train_base_path = "ALL_IDB2/img" , valid_base_path = 'validacao')
#base = BaseLoader('teste')
#base = BaseLoader('ALL_IDB2/img')
#base = BaseLoader('Teste_ALL_IDB2/V0')
#base = BaseLoader('Teste_ALL_IDB2/V0')
base = BaseLoader(train_base_path = 'ALL_IDB2/img' ,  valid_base_path = 'Teste_ALL_IDB2/ALL')
#base = BaseLoader(valid_base_path = 'Teste_ALL_IDB2/ALL')
#base = BaseLoader('teste_validacao')
base.load()


X = base.train_vectors
y = base.train_labels

#neigh = KNeighborsClassifier(n_neighbors=3)
#neigh.fit(X, y)
#classes = neigh.predict(base.valid_vectors)

#clf = SVC()
#clf.fit(X, y)
#classes = clf.predict(base.valid_vectors)

clf = LinearDiscriminantAnalysis()
clf.fit(X, y)
classes = clf.predict(base.valid_vectors)

corrects = 0
errors = 0
for index in xrange(0 , len(classes) - 1):
	if classes[index] == base.valid_labels[index]:
		corrects += 1
	else:
		errors += 1
print("Corrects : " + str(corrects))
print("ERRORS : " + str(errors))
percentage = (corrects * 100) / len(base.valid_vectors)
print("PERCENTAGE : " + str(percentage))


"""
for image in  base.train_images:
	segmented_image = Segmentation(image.path).process()
	cv2.imshow('segmented_image' , segmented_image)
	red , green , blue = ImageChanels(segmented_image).rgb()
	sum_image = red + green + blue
	print(image.path)
	show = np.concatenate(( red , green , blue , sum_image / 3) , axis=1)
	cv2.imshow('resault' , show)
	cv2.waitKey(350)
"""


"""
import matplotlib.pyplot as plt


values_1 = []
values_0 = []
for image in base.train_images:
	print("==============================================")
	print(image.path)
	segmented_image = Segmentation(image.path).process()
	value = FeatureExtractor(segmented_image).get_features()[3]
	#value = area_refs_min_circle(segmented_image)
	#cv2.imshow('image' , segmented_image)
	#cv2.waitKey(0)
	#value = get_cell_area(segmented_image)
	#value = get_cell_variance(segmented_image)
	#value = get_cell_perimeter(segmented_image)	
	if image.label == 1:
		values_1.append(value)
	else:
		values_0.append(value)
plt.plot(values_0 , 'go' , values_1 , 'rx')
plt.show()
"""
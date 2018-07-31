import cv2 
import os
import time

from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import preprocessing, neighbors, svm

def flat_to_one_hot(labels):
    num_classes = np.unique(labels).shape[0]
    num_labels = labels.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels,num_classes))
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot

train=pd.read_csv('/home/rishabh/train.csv')

#dividing data into labels and pixel values
X = train.ix[:,1:]
y = train.ix[:,0]

#normalizing
X = X.values/255.0
y = y.values.ravel()

accuracies=[]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)

#principal componant analysis(PCA)
pca = PCA(n_components=28, whiten = True, svd_solver='randomized') 
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)	

knn=[]
print '---------KNN----------'
#predicting for different values of K
neighbor_range=[1,2,3,4,5,6,7,8,9,10]
for i in neighbor_range:
	clf = neighbors.KNeighborsClassifier(n_neighbors=i)
	clf.fit(X_train_pca,y_train)
	accuracy = clf.score(X_test_pca,y_test)
	knn.append(accuracy)
	print ('Accuracy with k= %d is %f') % (i,accuracy)

fig=plt.figure()

#plotting the graph for accuracies from different values of K
axis1=fig.add_subplot(311)
axis1.plot(neighbor_range,knn,'r-o')
axis1.set_ylabel('accuracy')
axis1.set_xlabel('value of K')
axis1.set_title('Accuracies of K-Nearest Neighbours')

#calculating the max K-Nearest Neighbor accuracy
knn_max = max(knn)
accuracies.append(knn_max)

print '\n\n'
print '---------SVM----------'

#start = time.time()
  
print('SVC begin')
clf1 = svm.SVC(C=10)
clf1.fit(X_train_pca, y_train)

accuracy = clf1.score(X_test_pca, y_test)
accuracies.append(accuracy)

#end = time.time()
#print "accuracy: %f, time elaps: %f" % (accuracy, int(end-start))
print "Accuracy: %f" % (accuracy)


print '\n\n'
print '---------CNN----------'

y = flat_to_one_hot(y)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)
X_train_cnn = X_train.reshape(X_train.shape[0],1,28,28)
X_test_cnn = X_test.reshape(X_test.shape[0],1,28,28)

model = Sequential()
#hidden layer 1
model.add(Convolution2D(nb_filter=32,nb_row=5,nb_col=5,border_mode='same',input_shape=(1,28,28)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),border_mode='same'))

#hidden layer 2
model.add(Convolution2D(nb_filter=64,nb_row=5,nb_col=5,border_mode='same',input_shape=(32,14,14)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),border_mode='same'))

model.add(Flatten())
model.add(Dense(516))
model.add(Activation('relu'))	# relu - rectified linear unit
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

#adam - The method is also appropriate for non-stationary objectives and problems with very noisy and/or sparse gradients.
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train_cnn,y_train,batch_size=16,nb_epoch=1,verbose=1,validation_data=(X_test_cnn,y_test))
score = model.evaluate(X_test_cnn,y_test,verbose=0)
#y_pred = model.predict(X_test,batch_size=1,verbose=1)
accuracies.append(score[1])
print('Validation accuracy:', score[1])

# Read the input image 
im = cv2.imread("/home/rishabh/num.resized.jpg") 

# Convert to grayscale and apply Gaussian filtering 
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0) 

# Threshold the image 
ret, im_th = cv2.threshold(im_gray, 153, 255, cv2.THRESH_BINARY_INV) 

# Find contours in the image 
ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

# Get rectangles contains each contour 
rects = [cv2.boundingRect(ctr) for ctr in ctrs] 


prediction1=[]
for rect in rects: 
	# Draw the rectangles 
	cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 1) 

	# Make the rectangular region around the digit 
	leng = int(rect[3] * 1.6) 
	pt1 = int(rect[1] + rect[3] // 2 - leng // 2) 
	pt2 = int(rect[0] + rect[2] // 2 - leng // 2) 
	roi = im_th[pt1:pt1+leng, pt2:pt2+leng] 
	
	roi = im_th[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

	roi = cv2.dilate(roi, (3, 3)) 
	
	a = roi.shape[0]
	b = roi.shape[1] 
	#padding of matrix to make it a square matrix
	if a > b and (a-b) % 2 == 0:
		roi = np.pad(roi,[(0,0),((a-b)/2,(a-b)/2)], mode="constant", constant_values=0)
	elif a > b and (a-b) % 2 != 0:
		roi = np.pad(roi,[(0,0),((a-b)/2,((a-b)/2 + 1))], mode="constant", constant_values=0)
	elif a < b and (b-a) % 2 == 0:
		roi = np.pad(roi,[((b-a)/2,(b-a)/2),(0,0)], mode="constant", constant_values=0)
	elif a < b and (b-a) % 2 != 0:
		roi = np.pad(roi,[(((b-a)/2+1),(b-a)/2),(0,0)], mode="constant", constant_values=0)
	
	#resizing it to make 28x28
	roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
	#normalizing
	roi = roi/255.0

	roi=roi.reshape(1,1,28,28)

	prediction1 = model.predict(roi,batch_size=1,verbose=1)
	pred=np.argmax(prediction1)
	cv2.putText(im, str(pred), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1)
	print prediction1
	
#cv2.imshow("Resulting Image with Rectangular ROIs", im) 
cv2.imwrite('prediction.jpg',im)
cv2.waitKey()

#plotting the accuracies of the 3 methods used
axis2=fig.add_subplot(313)

my_ticks=['K-nn','SVM','CNN']
x=[1,2,3]
axis2.set_xticks(x)
axis2.set_xticklabels(my_ticks)
axis2.plot(x,accuracies,'-o')
axis2.set_ylabel('accuracy')
axis2.set_xlabel('methods')
axis2.set_title('Accuracy comparison of different methods')
fig.savefig('/home/rishabh/accuracy.jpeg')
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:37:36 2017

@author: thomas
"""

import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

#load and check data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print ('training data img shape :', train_images.shape, " label shape", train_labels.shape)
print ('test data img shape :', test_images.shape, " label shape", test_labels.shape)

#find the unique numbers from the train labels
classes = np.unique(train_labels)
nClasses = len(classes)
print('Total number of output: ', nClasses)
print('Output classes : ', classes)

plt.figure(figsize=[10,5])

#display the first image in training 
plt.subplot(121)
plt.imshow(train_images[0,:,:], cmap='gray')
plt.title("ground truth : {}".format(train_labels[0]))

#display the first image in testing data
plt.subplot(122)
plt.imshow(test_images[0,:,:], cmap='gray')
plt.title('Ground truth {}'.format(test_labels[0]))

##############################################################################
#Processing data

#Change from matrix to array of dimension 28*28 to array of dimension 784
dimData = np.prod(train_images.shape[1:1])
train_data = train_images.reshape(train_images.shape[0], dimData)
test_data = test_images.reshape(test_images.shape[0],dimData)

#change to float datatype
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

#scale the data to lie between 0 to 1
train_data /= 255
test_data /= 255

#Change the labels from integer to categorial data
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

#Display the change for category label using one-hot encoding
print('Original label 0 : ', train_labels[0])
print('After conversion to categorical (one-hot) : ', train_labels_one_hot[0])

##############################################################################
#Create the network model
#model = Sequential()
#model.add(Dense(512, activation = 'relu', input_shape = (dimData,)))
#model.add(Dense(512, activation = 'relu'))
#model.add(Dense(nClasses, activation = 'softmax'))

#on ajoute le dropout pour limiter le poids et donc l'overfitting
model_reg = Sequential()
model_reg.add(Dense(512, activation = 'relu', input_shape = (dimData,)))
model_reg.add(Dropout(0.5))
model_reg.add(Dense(512, activation = 'relu'))
model_reg.add(Dropout(0.5))
model_reg.add(Dense(nClasses, activation = 'softmax'))


##############################################################################
#Configure the model
model_reg.compile(optimize = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

##############################################################################
#Train the model
history = model_reg.fit(train_data, train_labels_one_hot, batch_size = 256, epochs = 20, verbose = 1,
                    validation_data = (test_data, test_labels_one_hot))

##############################################################################
#Evaluate on testing set
[test_loss, test_acc] =  model_reg.evaluate(test_data, test_labels_one_hot)
print("Evaluation result on Test_data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

##############################################################################
#Plot loss and accuracy on training and testing set

#Plot the loss Curves
plt.figure(figsize = [8,6])
plt.plot(history_reg.history['Loss'], 'r', linewidth = 3.0)
plt.plot(history_reg.history['val_loss'],)
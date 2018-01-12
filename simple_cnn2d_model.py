# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 23:21:50 2018

@author: aditya
"""

import config_v4 as cfg

import os
import numpy as np

from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D

from keras.datasets import cifar10

np.random.seed(1234)

input_neurons=200
input_dim=3072
act1='relu'
act2='relu'
act3='relu'
act4='softmax'
num_classes=len(cfg.labels)
dropout=0.2
filters = 32
kernel_size = 3




(train_x, train_y), (test_x, test_y) = cifar10.load_data()


#Checking Shapes
print "Train X shape",train_x.shape
print "Train Y shape",train_y.shape
print "Test X shape",test_x.shape
print "Test Y shape",test_y.shape

#train_y=to_categorical(train_y, num_classes=num_classes)
#bre


model = Sequential()

model.summary()
# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255
model.fit(train_x, train_y, epochs=10, batch_size=32,  verbose=1)



"""
# calculate predictions
y_pred = model.predict(test_x)
# round predictions
acc2=accuracy_score(test_y, y_pred)
print "Accuracy",acc2
"""
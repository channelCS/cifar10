# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 00:52:35 2018

@author: aditya
"""

from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.metrics import accuracy_score


from keras.utils import to_categorical

import numpy as np
np.random.seed(1234)


train_x=np.load('train_x.npy')
train_y=np.load('train_y.npy')
#test_x=np.load('test_x.npy')
#test_y=np.load('test_y.npy')
input_neurons=200
input_dim=3072
act1='relu'
act2='relu'
act3='relu'
act4='softmax'
num_classes=10
dropout=0.2
train_y=to_categorical(train_y, num_classes=None)

model = Sequential()

filters = 32
kernel_size = 3
"""
model.add(Dense(input_neurons, input_dim = input_dim, activation=act1))
model.add(Dropout(dropout))
model.add(Dense(input_neurons, activation=act2))
model.add(Dropout(dropout))
model.add(Dense(input_neurons, activation=act3))
model.add(Dropout(dropout))
model.add(Dense(num_classes, activation=act4))
#model.summary()
model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
   

 
# Fit the model
model.fit(train_x, train_y, epochs=10, batch_size=32,  verbose=1)
"""
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D
import os

model.add(Conv1D(filters, kernel_size,padding='same',
                 input_dim=input_dim))
#                 input_dim=train_x.shape[1]))
model.add(Activation('relu'))
model.add(Conv1D(filters,kernel_size))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))

model.add(Conv1D(filters*2,kernel_size, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(filters*2, 2))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))

#model.add(Flatten())
model.add(Dense(filters*4))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()
bre
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
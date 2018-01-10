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
act1='sigmoid'
act2='sigmoid'
act3='sigmoid'
act4='sigmoid'
num_classes=10
dropout=0.2

model = Sequential()
model.add(Dense(input_neurons, input_dim = input_dim, activation=act1))
model.add(Dropout(dropout))
model.add(Dense(input_neurons, activation=act2))
model.add(Dropout(dropout))
model.add(Dense(input_neurons, activation=act3))
model.add(Dropout(dropout))
model.add(Dense(num_classes, activation=act4))
#model.summary()
model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
   

train_y=to_categorical(train_y, num_classes=None)
 
# Fit the model
model.fit(train_x, train_y, epochs=10, batch_size=32,  verbose=2)



"""
# calculate predictions
y_pred = model.predict(test_x)
# round predictions
acc2=accuracy_score(test_y, y_pred)
print "Accuracy",acc2
"""
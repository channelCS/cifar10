# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 00:52:35 2018

@author: aditya
"""

from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

import numpy as np
np.random.seed(1234)


train_x=np.load('train_x.npy')
train_y=np.load('train_y.npy')
#test_x=np.load('test_x.npy')
#test_y=np.load('test_y.npy')

model = Sequential()
model.add(Dense(200, input_dim=3072, init='uniform', activation='relu'))
model.add(Dense(200, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(train_x, train_y, epochs=10, batch_size=32,  verbose=2)

"""
# calculate predictions
y_pred = model.predict(test_x)
# round predictions
acc2=accuracy_score(test_y, y_pred)
print "Accuracy",acc2
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 00:52:35 2018

@author: aditya
"""

from keras.models import Sequential
from keras.layers import Dense

import numpy as np
np.random.seed(1234)


X=np.load('train_x.npy')
Y=np.load('train_y.npy')

model = Sequential()
model.add(Dense(200, input_dim=3072, init='uniform', activation='relu'))
model.add(Dense(200, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=10, batch_size=32,  verbose=2)
# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)
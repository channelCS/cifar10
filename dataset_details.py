# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 23:11:19 2018

@author: aditya
"""


import csv
import config_v4 as cfg
import cPickle
import numpy as np
train_file=cfg.fe_orig_fd+'/data_batch_1'



with open(train_file, 'rb') as fo:
    dict_ = cPickle.load(fo)
    
print len(dict_)
j=0
arr=[]
arr2=[]
arr3=[]
for i in dict_:
    j+=1
    if j==1:
        arr.append(dict_[i])
    if j==2:
        arr2.append(dict_[i])
    if j==3:
        arr3.append(dict_[i])

arr=arr[0]
arr2=arr2[0]
train_x=np.array(arr)
train_y=np.array(arr2)
np.save('train_x.npy',train_x)
np.save('train_y.npy',train_y)
print "Unique elements in current batchfile",len(set(arr2))
"""
with open('def.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        print ', '.join(row)
"""




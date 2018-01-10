'''
SUMMARY:  config file
AUTHOR:   Aditya Arora
Created:  2018.01.10
Modified: NA
--------------------------------------
'''
# datset path
a = 'C:/Users/aditya/version-control/Datasets/cifar10'

fe_orig_fd 			 = a+'/Fe' #2
fe_eva_orig_fd 		 = a+'/Fe_eva' #2


meta_csv      = 'meta.txt'



# 1 of 10 image label
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

            
lb_to_id = { lb:id for id, lb in enumerate(labels) }
id_to_lb = { id:lb for id, lb in enumerate(labels) }

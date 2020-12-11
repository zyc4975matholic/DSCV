# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:17:50 2020

@author: ZYCBl
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sunday Nov 17 22:00:20 2019
@author: Wang, Zheng (zw1454@nyu.edu)
"""
import random
import numpy as np
import os
from sklearn.metrics import f1_score
import pandas as pd
from imblearn.metrics import geometric_mean_score as gms


uu_label = []      
uu_label_test = 99

# This code is implemented all by Yuchen Zhu. The code a part 1 code for running OSR baselines on COIL20
# For some reason, HPC can't deal with the tasks on COIL in a single file, so we have to split into 2 parts.
# First part is for model training
# This code is not meant for running.

uu = [6,10,13]
order = range(3)


def load_data(uu_num,idx):
    train = np.load("COIL20/{}-{} COIL20_train.npy".format(uu_num,idx),allow_pickle = True)
    test =  np.load("COIL20/{}-{} COIL20_test.npy".format(uu_num,idx),allow_pickle = True)
        
    batch = train.shape[0]
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    for i in range(batch):
                
        s_X_train,s_Y_train = np.hsplit(train[i], [-1])
        s_X_test,s_Y_test = np.hsplit(test[i], [-1])
            
        s_Y_train = s_Y_train.ravel()
        s_Y_test = s_Y_test.ravel()
        X_train.append(s_X_train)
        X_test.append(s_X_test)
        Y_train.append(s_Y_train)
        Y_test.append(s_Y_test)
    print("Finish Loading")
    
    return X_train,Y_train, X_test,Y_test



for uu_num in uu:
    gscore_lst = []
    f1_lst = []
    for j in order:
        x_train,y_train,x_test, y_test = load_data(uu_num,j)
        x_train,x_test,y_train,y_test = x_train[0], x_test[0], y_train[0], y_test[0]
        print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
        with open("COIL20_train_{}_{}.txt".format(uu_num,j),"w") as f:
            for idx in range(len(x_train)):
                f.write(str(int(y_train[idx])))
                for each in range(len(x_train[idx])):
                    f.write(" {}:{:.5f}".format(each,x_train[idx][each]))
                f.write("\n")
        with open("COIL20_test_{}_{}.txt".format(uu_num,j),"w") as f:
            for idx in range(len(x_test)):
                f.write(str(int(y_test[idx])))
                for each in range(len(x_test[idx])):
                    f.write(" {}:{:.5f}".format(each,x_test[idx][each]))
                f.write("\n")
            os.system("/gpfsnyu/home/yz4975/libsvm-openset/svm-train -s 8 -t 2 -q COIL20_train_{}_{}.txt train_{}_{}.model".format(uu_num,j,uu_num,j))# 1vssetmode
            # os.system("/gpfsnyu/home/yz4975/libsvm-openset/svm-predict -o -P 0.1 -C 0.001 COIL20_test.txt train.model result.txt")
            
            
  




# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 13:29:33 2020

@author: ZYCBl
"""

import random
import numpy as np


import os
from sklearn.metrics import f1_score
import pandas as pd
from imblearn.metrics import geometric_mean_score as gms

# This code is implemented all by Yuchen Zhu. The code a part 2 code for running OSR baselines on COIL20
# For some reason, HPC can't deal with the tasks on COIL in a single file, so we have to split into 2 parts.
# Second part is for gmean score computation.

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
        df = pd.read_csv("result_{}_{}.txt".format(uu_num,j), sep = ":" ,header = None)

        for i in df.index:
            if df.loc[i, 1] == 0:
                df.loc[i, 0] = 99
            
        y_predict = np.array(df[0])
        gscore = gms(y_test, y_predict, average = "weighted")
        f_measure = f1_score(y_test, y_predict, average='weighted')
        gscore_lst.append(gscore)
        f1_lst.append(f_measure)
        
                
    gs_mean = np.array(gscore_lst).mean()
    f1_mean = np.array(f1_lst).mean()
    with open("score.txt","a") as f:
        f.write("COIL\t{}\t{:.4f}\t{:.4f}\n".format(uu_num,f1_mean,gs_mean))

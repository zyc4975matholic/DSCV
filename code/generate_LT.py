# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 21:18:26 2020

@author: ZYCBl
"""

import random
import numpy as np


from sklearn.model_selection import train_test_split


import pandas as pd
from collections import Counter
from imblearn.datasets import make_imbalance
from math import floor
from scipy.stats import pareto

uu_label_test = 99

# This code is implemented all by Yuchen Zhu. The code is written for generating Long-tailed version of real world dataset.
# This code is not meant for running unless provided input file correctly. 



def make_data_letter(uu_label,idx):
    # Generate LETTER-LT
    input_data = pd.read_csv('LETTER.csv').to_numpy()
    x = input_data[..., 1:]
    y = input_data[..., 0]

    total = 26
    unseen = len(uu_label)
    all_num = total - unseen
    wt = np.linspace(1, ratio ** 0.5, all_num)

    for i in range(y.shape[0]):
        y[i] = ord(y[i]) % 32
        if y[i] in uu_label:
            y[i] = uu_label_test
    
    old_x_train, x_test, old_y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train,y_train = old_x_train[old_y_train != uu_label_test], old_y_train[old_y_train != uu_label_test]
    

    c = Counter(y_train)
    z = [i for i in range(1,27) if i not in uu_label]
    
    size = list(pareto.pdf(wt, b = 1))
    random.shuffle(size)

    strategy = {i: floor(c[i] * size.pop()) for i in z}
    y_train = y_train.astype("int")

    x_res, y_res = make_imbalance(x_train,y_train, sampling_strategy = strategy)

    zipped = list(zip(x_test, y_test))  
    random.shuffle(zipped)
    x_test, y_test = zip(*zipped)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    
    
    train_list = []
    test_list = []
        
    batch = 1    
    for i in range(batch):
        X_train = x_res
        Y_train = y_res
        X_test = x_test
        Y_test = y_test
        Y_train = Y_train.reshape(-1,1)
        Y_test = Y_test.reshape(-1,1)
        train = np.hstack((X_train,Y_train))
        test = np.hstack((X_test,Y_test))
            
        train_list.append(train)
        test_list.append(test)
        train_list = np.array(train_list)
        test_list = np.array(test_list)
    np.save("pendigits/{}-{} letter_train.npy".format(len(uu_label),idx), train_list)
    np.save("pendigits/{}-{} letter_test.npy".format(len(uu_label),idx), test_list)


def make_data_pen(uu_label,idx):
    # Generate PENDIGITS-LT
    input_data = pd.read_csv('PENDIGITS.csv').to_numpy()
    x = input_data[..., 1:]
    y = input_data[..., 0]
    print(y.shape)

    total = 10
    unseen = len(uu_label)
    all_num = total - unseen
    wt = np.linspace(1, ratio ** 0.5, all_num)

    y = y.astype(int)
    for uu in uu_label:
        y = np.where(y == uu, 99 ,y)
    
    old_x_train, x_test, old_y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train,y_train = old_x_train[old_y_train != uu_label_test], old_y_train[old_y_train != uu_label_test]
    

    c = Counter(y_train)
    z = [i for i in range(0,10) if i not in uu_label]
    
    size = list(pareto.pdf(wt, b = 1))
    random.shuffle(size)

    strategy = {i: floor(c[i] * size.pop()) for i in z}
    y_train = y_train.astype("int")
    
    x_res, y_res = make_imbalance(x_train,y_train, sampling_strategy = strategy)

    zipped = list(zip(x_test, y_test))  
    random.shuffle(zipped)
    x_test, y_test = zip(*zipped)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    
    
    train_list = []
    test_list = []
        
    batch = 1    
    for i in range(batch):
        X_train = x_res
        Y_train = y_res
        X_test = x_test
        Y_test = y_test
        Y_train = Y_train.reshape(-1,1)
        Y_test = Y_test.reshape(-1,1)
        train = np.hstack((X_train,Y_train))
        test = np.hstack((X_test,Y_test))
            
        train_list.append(train)
        test_list.append(test)
        train_list = np.array(train_list)
        test_list = np.array(test_list)
    np.save("pendigits/{}-{} pendigits_train.npy".format(len(uu_label),idx), train_list)
    np.save("pendigits/{}-{} pendigits_test.npy".format(len(uu_label),idx), test_list)



def make_data_COIL(uu_label,idx):
    # Generates COIL20-LT
    input_data = pd.read_csv('COIL20.csv').to_numpy()
    x = input_data[..., 1:]
    y = input_data[..., 0]
    print(y.shape)
    print(x[0],y[0])
    total = 20
    unseen = len(uu_label)
    all_num = total - unseen
    wt = np.linspace(1, ratio ** 0.5, all_num)

    y = y.astype(int)
    for uu in uu_label:
        y = np.where(y == uu, 99 ,y)
    
    old_x_train, x_test, old_y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train,y_train = old_x_train[old_y_train != uu_label_test], old_y_train[old_y_train != uu_label_test]
    

    c = Counter(y_train)
    z = [i for i in range(1,20) if i not in uu_label]
    
    size = list(pareto.pdf(wt, b = 1))
    random.shuffle(size)

    strategy = {i: floor(c[i] * size.pop()) for i in z}
    y_train = y_train.astype("int")
    
    x_res, y_res = make_imbalance(x_train,y_train, sampling_strategy = strategy)

    zipped = list(zip(x_test, y_test))  
    random.shuffle(zipped)
    x_test, y_test = zip(*zipped)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    
    
    train_list = []
    test_list = []
        
    batch = 1    
    for i in range(batch):
        X_train = x_res
        Y_train = y_res
        X_test = x_test
        Y_test = y_test
        Y_train = Y_train.reshape(-1,1)
        Y_test = Y_test.reshape(-1,1)
        train = np.hstack((X_train,Y_train))
        test = np.hstack((X_test,Y_test))
            
        train_list.append(train)
        test_list.append(test)
        train_list = np.array(train_list)
        test_list = np.array(test_list)
    np.save("COIL20/{}-{} COIL20_train.npy".format(len(uu_label),idx), train_list)
    np.save("COIL20/{}-{} COIL20_test.npy".format(len(uu_label),idx), test_list)
# total_uu = [[20,13,12,5,1,3],[16,2,17,14,4,5], [10,4,18,8,15,19]]
# total_uu = [[5,13,12,3,11,16,17,7,2,1],[6,11,13,15,1,20,3,16,2,18],[13,5,1,12,9,4,11,14,8,10]]
total_uu = [[5,14,12,3,11,16,10,7,2,1,20,18,19],[6,11,13,15,1,8,3,19,2,18,0,5,7],[19,5,1,12,15,4,11,14,8,10,2,7,20]]
ratio = 2
for j in range(3):
    make_data_COIL(total_uu[j],j)

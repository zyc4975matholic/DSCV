# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 19:05:46 2020

@author: ZYCBl
"""
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
import pandas as pd
from time import time
from imblearn.metrics import geometric_mean_score as gms



#This code is implemented all by Yuchen Zhu. The code is for running OSR methods on PENDIGITS-LT
#This code is not meant for running.

uu_label = [[1,3,5],[7,8,0,5,3],[2,4,6],[7,9,10],[3,5,8],[1,7,9], [7,1,8,2,3], [7,6,8,0,2],[5,1,7,6,3], [4,7,8,6,9]]
uu_label_test = 99

'''------------------ make data ------------------------ 
In this part, we generate our data and perform
a multidimensional scaling (MDS) to visualize the class separability.
Then we split our generated data into test/training sets.
'''

def make_data(uu_label):
    
    input_data = pd.read_csv('PENDIGITS.csv').values       

    x = input_data[..., 1:]
    y = input_data[..., 0]
    
    for i in range(y.shape[0]):
        if y[i] in uu_label:
            y[i] = uu_label_test
    
    old_x_train, x_test, old_y_train, y_test = train_test_split(x, y, test_size=0.4)

    x_train = []
    y_train = []
    uu_x = []
    uu_y = []
    for i in range(old_x_train.shape[0]):
        if old_y_train[i] != uu_label_test:
            x_train.append(old_x_train[i])
            y_train.append(old_y_train[i])
        else:
            uu_x.append(old_x_train[i])
            uu_y.append(old_y_train[i])
       
    x_train = np.array(x_train)     
    y_train = np.array(y_train)
    uu_x = np.array(uu_x)
    uu_y = np.array(uu_y)
    
    print("Shape of all data: ", x.shape)
    print("Shape of unknown unknown data: ", uu_x.shape)
    
    zipped = list(zip(x_test, y_test))  
    random.shuffle(zipped)
    x_test, y_test = zip(*zipped)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
#    print("Shape of the original training data: ", x_train.shape)
#    print("Shape of the original test data: ", x_test.shape)
    
    '''----------------------------------------------------'''    
#    print("Performing Multidimensional Scaling...")
#    embedding = MDS(n_components=2)
#    x_transformed = embedding.fit_transform(x)
#    
#    pyplot.figure(1)
#    pyplot.scatter(x_transformed[:,0], x_transformed[:,1],c=y)
#    pyplot.title("The structure of the iris classes")
#    pyplot.show()
    '''----------------------------------------------------'''
    
    return x_train, x_test, y_train, y_test



def load_data(uu_num,idx):
    train = np.load("pendigits/{}-{} pendigits_train.npy".format(uu_num,idx),allow_pickle = True)
    test =  np.load("pendigits/{}-{} pendigits_test.npy".format(uu_num,idx),allow_pickle = True)
        
    train = train.astype("int")
    test = test.astype("int")
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


uu = [3,5,7]
order = range(3)


for uu_num in uu:
    gscore_lst = []
    f1_lst = []
    for idx in order:
        x_train,y_train,x_test, y_test = load_data(uu_num,idx)
        x_train,x_test,y_train,y_test = x_train[0], x_test[0], y_train[0], y_test[0]
        print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
        with open("PENDIGITS_train.txt","w") as f:
            for idx in range(len(x_train)):
                f.write(str(int(y_train[idx])))
                for each in range(len(x_train[idx])):
                    f.write(" {}:{:2d}".format(each,x_train[idx][each]))
                f.write("\n")
        with open("PENDIGITS_test.txt","w") as f:
            for idx in range(len(x_test)):
                f.write(str(int(y_test[idx])))
                for each in range(len(x_test[idx])):
                    f.write(" {}:{:2d}".format(each,x_test[idx][each]))
                f.write("\n")
        os.system("/gpfsnyu/home/yz4975/libsvm-openset/svm-train -s 8 -t 2 -q PENDIGITS_train.txt train.model") #1vsset mode
        os.system("/gpfsnyu/home/yz4975/libsvm-openset/svm-predict -o -P 0.1 -C 0.001 PENDIGITS_test.txt train.model result.txt")
        
    
        df = pd.read_csv("result.txt", sep = ":" ,header = None)

        for i in df.index:
            if df.loc[i, 1] == 0:
                df.loc[i, 0] = 99
        y_predict = np.array(df[0])
        gscore = gms(y_test, y_predict, average = "weighted")
        f_measure = f1_score(y_test, y_predict, average='weighted')
        gscore_lst.append(gscore)
        f1_lst.append(f_measure)

        os.system("rm result.txt")
        os.system("rm train.model")
        os.system("rm PENDIGITS_train.txt")
        os.system("rm PENDIGITS_test.txt")
    gs_mean = np.array(gscore_lst).mean()
    f1_mean = np.array(f1_lst).mean()
    with open("score.txt","a") as f:
        f.write("PEN\t{}\t{:.4f}\t{:.4f}\n".format(uu_num,f1_mean,gs_mean))
        
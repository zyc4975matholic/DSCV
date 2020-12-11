# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 17:34:35 2020

@author: ZYCBl
"""

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
import os
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
import pandas as pd
from imblearn.metrics import geometric_mean_score as gms




uu_label = [[2,26,8,19,14,17,25,23,10,16,5]]
uu_label_test = 99

'''------------------ make data ------------------------ 
In this part, we generate our data and perform
a multidimensional scaling (MDS) to visualize the class separability.
Then we split our generated data into test/training sets.
'''

def make_data(uu_label):
    
    input_data = pd.read_csv('LETTER.csv').values      

    x = input_data[..., 1:]
    y = input_data[..., 0]
    
    for i in range(y.shape[0]):
        y[i] = ord(y[i]) % 32
        if y[i] in uu_label:
            y[i] = uu_label_test
    
    
    
    old_x_train, x_test, old_y_train, y_test = train_test_split(x, y, test_size=0.3)

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
    train = np.load("letter/{}-{} letter_train.npy".format(uu_num,idx),allow_pickle = True)
    test =  np.load("letter/{}-{} letter_test.npy".format(uu_num,idx),allow_pickle = True)
        
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


uu = [5,11,16]
order = range(3)


for uu_num in uu:
    gscore_lst = []
    f1_lst = []
    for idx in order:
        x_train,y_train,x_test, y_test = load_data(uu_num,idx)
        x_train,x_test,y_train,y_test = x_train[0], x_test[0], y_train[0], y_test[0]
        print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
        with open("LETTER_train.txt","w") as f:
            for idx in range(len(x_train)):
                f.write(str(int(y_train[idx])))
                for each in range(len(x_train[idx])):
                    f.write(" {}:{:2d}".format(each,x_train[idx][each]))
                f.write("\n")
        with open("LETTER_test.txt","w") as f:
            for idx in range(len(x_test)):
                f.write(str(int(y_test[idx])))
                for each in range(len(x_test[idx])):
                    f.write(" {}:{:2d}".format(each,x_test[idx][each]))
                f.write("\n")
        os.system("/gpfsnyu/home/yz4975/libsvm-openset/svm-train -s 8 -t 2 -q LETTER_train.txt train.model") # 1vssetmode
        os.system("/gpfsnyu/home/yz4975/libsvm-openset/svm-predict -o -P 0.1 -C 0.001 LETTER_test.txt train.model result.txt")
        
        
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
                
    gs_mean = np.array(gscore_lst).mean()
    f1_mean = np.array(f1_lst).mean()
    with open("score.txt","a") as f:
        f.write("LETTER\t{}\t{:.4f}\t{:.4f}\n".format(uu_num,f1_mean,gs_mean))
        
    
    


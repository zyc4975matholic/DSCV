# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 04:05:42 2020

@author: ZYCBl
"""
import random
import numpy as np
import seaborn as sns

from numpy.random import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from collections import Counter
from imblearn.under_sampling import *
from imblearn.over_sampling import *
from imblearn.combine import *
from sklearn.base import BaseEstimator
from sklearn.datasets import load_wine
from sklearn.metrics import f1_score

# This code is implemented all by Yuchen Zhu. This code implements the framework DSCV. See more information in Readme.md
# This code is for running. Run the code directly for testing.

class DSCV(BaseEstimator):
    # Pass in an instance of base classfier and resampler type and the keyword arguments(using dictionary), default is SVM, None and SMOTE
    
'''
    To intialize a DSCV instance the following parameter has to be passed
        classifier: A classifier object( Not instance) that at least implement fit and predict methods, default = SVC
        kwargs: A dictionary that contains the parameter-value pairs used to generate the classifier, default = default setting of SVC
        sample_rate: A float between 0 and 1, indicates the test sample size, default = 0.2
        k_fold: A int larger than 2, indicates the number of folds in cross validation, default = 3
        resampler: A str that indicates the sampling strategy. Must be one of the following, default = "SMOTE"
            "ClusterCentroids"
            "CondensedNearestNeighbour"
            "EditedNearestNeighbours"
            "RepeatedEditedNearestNeighbours"
            "AllKNN"
            "InstanceHardnessThreshold"
            "NearMiss"
            "NeighbourhoodCleaningRule"
            "OneSidedSelection"
            "RandomUnderSampler"
            "TomekLinks"
            "SVM-SMOTE"
            "ADASYN"
            "KMeansSMOTE"
            "BorderlineSMOTE"
            "RandomOverSampler"
            "SMOTE"
            "SMOTEENN"
            "SMOTETomek"
     Trainning with DSCV
        use method meta_fit, passing in X_train, Y_train, X_test, Y_test return a trained model with DSCV
     Predict with DSCV
        use method predict, passing X, return a numpy array with predicted labels, label with 99 are unknown unknowns found by DSCV
'''
    
    
    def __init__(self, classifier = SVC, kwargs = {"C":1, "kernel":"rbf"}, resampler = "SMOTE", sample_rate = 0.2, k_fold = 3):
        self.classifier = classifier
        self.resampler = resampler
        self.kwargs = kwargs
        self.sample_rate = sample_rate
        self.k_fold = k_fold
        
    def J1_score(self, size, k_mean,k_cov, uu_mean, dim):
        # J_1 socre is a metric to evaluate the class separability of a dataset
        # In this function, k_J_1 is the known class separability, which only considers the known class in the trainning set, also the computation follows
        # the tradional definition of J1 score
        # u_J_1 is the unknown class separability, which is a modified version of the definition, consider both known class and unknown class
        
        mean = uu_mean.mean(axis = 0) # mean_overall of unknown class
        mean = mean.reshape(dim,1)
        u_S_B = np.sum([size[i] * (k_mean[i] - mean).dot((k_mean[i] - mean).T) for i in range(k_mean.shape[0])], axis = 0)
        u_s_b = np.trace(u_S_B)
        k_S_W = np.sum([size[i] * k_cov[i] for i in range(k_cov.shape[0])],axis = 0)
        k_s_w = np.trace(k_S_W)
        k_overall_mean = k_mean.mean(axis = 0)
        k_overall_mean = k_overall_mean.reshape(dim,1)
        k_S_B = np.sum([size[i] * (k_mean[i] - k_overall_mean).dot((k_mean[i] -k_overall_mean).T) for i in range(k_mean.shape[0])],axis = 0)
        k_s_b = np.trace(k_S_B)
        u_J_1 = 1 + u_s_b/ k_s_w
        k_J_1 = 1 + k_s_b/ k_s_w
        
        return k_J_1, u_J_1  # Return known class J1 score and Unknown class J1 score
    
    def J1_estimate(self,X_train,Y_train, X_test, Y_test):
        
        # Estimated J1 score from data
        train_label = set(Y_train)
        known_mean = []
        known_cov = []
        size = []
        dim = X_train.shape[1]
        n = len(X_train)
        for each in train_label:
            this_class = X_train[Y_train == each]
            this_size = len(this_class)/ n
            this_mean = this_class.mean(axis = 0)
            this_cov = np.cov(this_class, rowvar = False)
            known_mean.append(this_mean)
            known_cov.append(this_cov)
            size.append(this_size)
            
            
        uu = X_test[Y_test == 99]
        uu_mean = [uu.mean(axis = 0)]
        
        known_mean = np.array(known_mean)
        known_cov = np.array(known_cov)
        uu_mean = np.array(uu_mean)
        
        return self.J1_score(size, known_mean, known_cov, uu_mean, dim)

        
    def random_sampling(self,sample_size = 0.2):
        # Create a test sample
        # sample_size:  the percentage of sample in the test set
        x_test = self.X_test
        y_test = self.Y_test
        x_new_test, sample, y_new_test, sample_label = train_test_split(x_test,y_test, test_size = sample_size)
    
        return sample, sample_label, x_new_test, y_new_test
    
    def cross_validation(self, X,Y, k = 3):
        # Cross validation in RTSCV to predict Unknown classes
        # k  number of folds, doesn't affect the result a lot in a balanced setting, set to a lower value to save computational power
        kf = KFold(n_splits=3)
        uu_count = 0
        uu_total = []
        for train_index, test_index in kf.split(X, Y):
            xc_train, xc_test = X[train_index], X[test_index]
            yc_train, yc_test = Y[train_index], Y[test_index]
            cross_model = self.classifier(**self.kwargs)
            cross_model.fit(xc_train, yc_train)
        # sort out the samples classified into the new class
            pred = cross_model.predict(xc_test)
            uu = xc_test[pred == 99]
            uu_count  += len(uu)
            uu_total.append(uu)
            
        unknown = np.concatenate(uu_total)
        unknown_label = 99* np.ones((len(unknown),))
        
        return unknown, unknown_label
    
    
    def phase2(self, x_train, y_train):
        # RTSCV training
        
        x_train2 = x_train
        y_train2 = y_train
        
        sample, sample_label, x_test,y_test = self.random_sampling(sample_size = self.sample_rate)
        new_label = 99* np.ones((len(sample_label,)))
        x_train,y_train = np.concatenate([x_train,sample]),np.concatenate([y_train,new_label])
        
        zipped = list(zip(x_train,y_train))  
        random.shuffle(zipped)
        x_train,y_train = zip(*zipped)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
    
        un_class, un_label = self.cross_validation(x_train,y_train, k = self.k_fold)
    # construct the final training set by adding the unknown unknowns
   
        print("Shape of Mined Unknown class:", un_class.shape)
        if un_class.shape[0] != 0:
            x_train2 = np.concatenate((x_train2, un_class), axis=0)
        else:
            x_train2 = x_train2
        y_train2 = np.append(y_train2, un_label)
#     # train & test on the new sets
        model2 = self.classifier(**self.kwargs)
        model2.fit(x_train2, y_train2)
        
        self.model = model2       
        return self
    
    
    def meta_fit(self,X_train,Y_train, X_test,Y_test):
        resampler_dict = dict([("ClusterCentroids", ClusterCentroids(n_jobs = -1)),
                    ("CondensedNearestNeighbour", CondensedNearestNeighbour(n_jobs = -1)),
                    ("EditedNearestNeighbours", EditedNearestNeighbours(n_jobs = -1)),
                    ("RepeatedEditedNearestNeighbours", RepeatedEditedNearestNeighbours(n_jobs = -1)),
                    ("AllKNN", AllKNN(n_jobs = -1)),
                    ("InstanceHardnessThreshold", InstanceHardnessThreshold(n_jobs = -1)),
                    ("NearMiss", NearMiss(n_jobs = -1)),
                    ("NeighbourhoodCleaningRule", NeighbourhoodCleaningRule(n_jobs = -1)),
                    ("OneSidedSelection", OneSidedSelection(n_jobs = -1)),
                    ("RandomUnderSampler", RandomUnderSampler()),
                    ("TomekLinks", TomekLinks(n_jobs = -1)),
                    ("SVM-SMOTE", SVMSMOTE(n_jobs = -1)),
                    ("ADASYN",ADASYN(n_jobs = -1)),
                    ("KMeansSMOTE", KMeansSMOTE(n_jobs = -1)),
                    ("BorderlineSMOTE", BorderlineSMOTE(n_jobs = -1)),
                    ("RandomOverSampler", RandomOverSampler()),
                    ("SMOTE", SMOTE(n_jobs = -1)),
                    ("SMOTEENN", SMOTEENN(n_jobs = -1)),
                    ("SMOTETomek", SMOTETomek(n_jobs = -1))])
        
        self.X_train,self.Y_train,self.X_test,self.Y_test = X_train,Y_train, X_test,Y_test
        try:
            resampler = resampler_dict[self.resampler]
        except KeyError:
            raise Exception("Resampling Strategy Not Supported")

        print('Distribution before resampling: {}'.format(Counter(self.Y_train)))
        try:
            self.X_res,self.Y_res = resampler.fit_resample(self.X_train, self.Y_train)
        except:
            raise Exception("Resampling Process Failed. Try Other Sampling Strategy Or Check The Data")
        print('Resampled dataset shape %s' % Counter(self.Y_res))
        
        return self.phase2(self.X_res, self.Y_res)
    
    def predict(self,X):
        return self.model.predict(X)


       
    
if __name__ == '__main__': 
    
    # A test example 
    A = DSCV(sample_rate = 0.3)
    
    mat_cov = np.array([[0.1,0],[0,0.1]])
    mu1 = np.array([-2,0])
    mu2 = np.array([2,0])
    mu3 = np.array([0,-3])
    
    class1 = multivariate_normal(mu1,mat_cov,50)  #We treat class 1,2 as known calss
    class2 = multivariate_normal(mu2,mat_cov,50)
    class3 = multivariate_normal(mu3,mat_cov,50)  #We treat class 3 as uu
    
    label1 = np.ones(50)
    label2 = 2*np.ones(50)
    label3 = 99*np.ones(20)
    X = np.concatenate([class1,class2])
    y = np.concatenate([label1,label2])
    
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2)
    X_test = np.concatenate([X_test,class3])
    y_test = np.concatenate([y_test,label3])
    

    zipped = list(zip(X_test,y_test))  
    random.shuffle(zipped)
    X_test,y_test = zip(*zipped)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    A.meta_fit(X_train,y_train,X_test,y_test)
    y_predict = A.predict(X_test)
    print(f1_score(y_predict,y_test, average = "macro"))
    

    

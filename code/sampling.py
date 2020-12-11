# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 02:56:01 2020

@author: ZYCBl
"""
import random
import numpy as np
from RTSCV import RTSCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from scipy.stats import pareto
import warnings
import pandas as pd
from numpy.random import multivariate_normal
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin
from collections import Counter
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from math import ceil
from imblearn.datasets import make_imbalance
from imblearn.under_sampling import *
from imblearn.over_sampling import *
from imblearn.combine import *
from imblearn.ensemble import *
from imblearn.metrics import geometric_mean_score as gms
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_unif

from sklearn.multiclass import OneVsRestClassifier
import canonical_ensemble as ce
import self_paced_ensemble as spe
import ramo

#
# This code is implemented all by Yuchen Zhu. The code is written for running experiments on DSCV with different sampling strategy
# This code is not meant for running unless provided training and testing set properly 
# Call under_test to test undersampling methods, call over_test to test oversampling and hybrid sampling methods

    
class sample_stategy(RTSCV):
    
    def __init__(self):
        super().__init__()
        self.under_sampler = [("ClusterCentroids", ClusterCentroids(n_jobs = -1)),
                    ("CondensedNearestNeighbour", CondensedNearestNeighbour(n_jobs = -1)),
                    ("EditedNearestNeighbours", EditedNearestNeighbours(n_jobs = -1)),
                    ("RepeatedEditedNearestNeighbours", RepeatedEditedNearestNeighbours(n_jobs = -1)),
                    ("AllKNN", AllKNN(n_jobs = -1)),
                    ("InstanceHardnessThreshold", InstanceHardnessThreshold(n_jobs = -1)),
                    ("NearMiss", NearMiss(n_jobs = -1)),
                    ("NeighbourhoodCleaningRule", NeighbourhoodCleaningRule(n_jobs = -1)),
                    ("OneSidedSelection", OneSidedSelection(n_jobs = -1)),
                    ("RandomUnderSampler", RandomUnderSampler()),
                    ("TomekLinks", TomekLinks(n_jobs = -1))]
    
        self.over_sampler = [("SVM-SMOTE", SVMSMOTE(n_jobs = -1)),
                    ("ADASYN",ADASYN(n_jobs = -1)),
                    ("KMeansSMOTE", KMeansSMOTE(n_jobs = -1)),
                    ("BorderlineSMOTE", BorderlineSMOTE(n_jobs = -1)),
                    ("RandomOverSampler", RandomOverSampler()),
                    ("SMOTE", SMOTE(n_jobs = -1)),
                    ("SMOTEENN", SMOTEENN(n_jobs = -1)),
                    ("SMOTETomek", SMOTETomek(n_jobs = -1))]

        
        
    def under_test(self):
        im_ratio = list(range(1,200,5)) + [200]
        sep_condition = [1,4,7,10]

        sampler = self.under_sampler

        

        # with open("undersampling_strategy.txt", "a") as f:
        #     f.write("Strategy\timbalanced_ratio\tclass_separability\tk_J1\tu_J1\tcv_score\timprove_score\tcv_gms\timprove_gms\n")
        
        
        for resampler in sampler:
            for sep in sep_condition:
                for ratio in im_ratio:
                    X_train,Y_train, X_test,Y_test = self.load_data(ratio = ratio, sep = sep)
                    batch = len(X_train)

                    
                    for i in range(batch):
                        self.X_train,self.Y_train,self.X_test,self.Y_test = X_train[i],Y_train[i], X_test[i],Y_test[i]
                        
                        k_J1,u_J1 = self.J1_estimate(self.X_train,self.Y_train,self.X_test,self.Y_test)
                        
                        f1_cv,g_cv = self.phase2(self.X_train, self.Y_train)
                        if ratio == 1:
                            f1_imp,g_imp = f1_cv,g_cv
                        else:
                            try:
                                f1_imp,g_imp = self.resample(resampler[1])
                                
                            except:
                                f1_imp = np.float64("nan")
                                g_imp = np.float64("nan")
                       
                        with open("undersampling_strategy.txt", "a") as f:
                            f.write("{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(resampler[0], ratio,sep,k_J1, u_J1, f1_cv, f1_imp, g_cv, g_imp))
                        
            with open("undersampling_strategy.txt", "a") as f:
                f.write("\n")
                
    def over_test(self):
        # im_ratio = [10,100,200]
        # sep_condition = [1,4,8]
        im_ratio = list(range(1,200,5)) + [200]
        sep_condition = [1,4,7,10]

        # im_ratio = [10]
        # sep_condition = [1]
        
        sampler = self.over_sampler

        # with open("oversampling_strategy.txt", "a") as f:
        #     f.write("Strategy\timbalanced_ratio\tclass_separability\tk_J1\tu_J1\tcv_score\timprove_score\tcv_gms\timprove_gms\n")
        
        
        for resampler in sampler:
            for sep in sep_condition:
                for ratio in im_ratio:
                    X_train,Y_train, X_test,Y_test = self.load_data(ratio = ratio, sep = sep)
                    batch = len(X_train)

                    
                    for i in range(batch):
                        
                        self.X_train,self.Y_train,self.X_test,self.Y_test = X_train[i],Y_train[i], X_test[i],Y_test[i]
                        
                        k_J1,u_J1 = self.J1_estimate(self.X_train,self.Y_train,self.X_test,self.Y_test)
                        
                        f1_cv,g_cv = self.phase2(self.X_train, self.Y_train)
                        if ratio == 1:
                            f1_imp,g_imp = f1_cv,g_cv
                            
                        else:
                            try:
                                f1_imp,g_imp = self.resample(resampler[1])
                                

                            except:
                                f1_imp = np.float64("nan")
                                g_imp = np.float64("nan")
                        

                        
                        with open("oversampling_strategy.txt", "a") as f:
                            f.write("{}\t{}\t{}\t{:.2f}\t{:.2f}\t {:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(resampler[0], ratio,sep,k_J1, u_J1, f1_cv, f1_imp, g_cv, g_imp))

            with open("oversampling_strategy.txt", "a") as f:
                f.write("\n")
    
    def benchmark(self):
        im_ratio = [1,10,50,100]
        sep_condition = [1,3,5,7,10]
        
        
        with open("balanced_benchmark.txt", "a") as f:
            f.write("imbalanced_ratio\tclass_separability\tk_J1\tu_J1\tcv_score\traw_score\tcv_gms\traw_gms\n")
        
        
        for sep in sep_condition:
            for ratio in im_ratio:
                X_train,Y_train, X_test,Y_test = self.load_data(ratio = ratio, sep = sep)
                batch = len(X_train)

                for i in range(batch):
                    self.X_train,self.Y_train,self.X_test,self.Y_test = X_train[i],Y_train[i], X_test[i],Y_test[i]
                        
                    k_J1,u_J1 = self.J1_estimate(self.X_train,self.Y_train,self.X_test,self.Y_test)
                    
                    f1_raw,g_raw = self.phase1()
                    
                    f1_cv,g_cv = self.phase2(self.X_train, self.Y_train)
                    
                    with open("balanced_benchmark.txt", "a") as f:
                        f.write("{}\t{}\t{:.2f}\t{:.2f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(ratio,sep,k_J1, u_J1, f1_cv, f1_raw, g_cv, g_raw))
                        
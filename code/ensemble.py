# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 02:59:55 2020

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


# This code is implemented all by Yuchen Zhu. The code is written for running experiments on RTSCV with different ensemble calssifier
# implemented in Imbalanced-learn
# This code is not meant for running unless provided training and testing set properly 
# Call test to test the ensemble methods



                
class ensemble(RTSCV):
        
    def cross_validation(self, X,Y, k = 3):
        # Cross validation in RTSCV to predict Unknown classes
        # k  number of folds, doesn't affect the result a lot in a balanced setting, set to a lower value to save computational power
        
        kf = KFold(n_splits=k)
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
    
    
    def phase2(self, x_train = None,y_train = None):
        # RTSCV training
        

        x_train2 = x_train
        y_train2 = y_train
        
        sample, sample_label, x_test,y_test = self.random_sampling(sample_size = 0.2)
        new_label = 99* np.ones((len(sample_label,)))
        x_train,y_train = np.concatenate([x_train,sample]),np.concatenate([y_train,new_label])
        
        zipped = list(zip(x_train,y_train))  
        random.shuffle(zipped)
        x_train,y_train = zip(*zipped)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        
        
        
        
        pipeline = Pipeline([('test', self.classifier())])
        # Parameter distributions
        params_dict = self.params_dict
        #Search randomly over parameter space
        random_search = RandomizedSearchCV(
                pipeline,
                params_dict,
                n_iter  =50,
                scoring ="f1_weighted",
                verbose =1,
                cv      =5,
                n_jobs = -1,
                return_train_score=True)

        random_search.fit(x_train,y_train)
        
        
        
        params = random_search.best_params_
        new_params = dict()
        for k,v in params.items():
            new_params[k[6:]] = v
            
        self.kwargs = new_params
        print(self.kwargs)
        
        
    
        un_class, un_label = self.cross_validation(x_train,y_train)
    
    # construct the final training set by adding the unknown unknowns
   
        print("Shape of Mined Unknown class:", un_class.shape)
        if un_class.shape[0] != 0:
            x_train2 = np.concatenate((x_train2, un_class), axis=0)
        else:
            x_train2 = x_train2
        y_train2 = np.append(y_train2, un_label)

#     # train & test on the new sets
        model2 = SVC(C=1.0, cache_size=200, class_weight="balanced", coef0=0.0,
                     decision_function_shape='ovr', degree=3, gamma='auto',
                     kernel='rbf', max_iter=-1, probability=False, random_state=None,
                     shrinking=True, tol=0.001, verbose=False)
        model3 = self.classifier(**self.kwargs)
    
        model2.fit(x_train2, y_train2)
        model3.fit(x_train2,y_train2)
        
        self.model = model2
        
        origin_pred = model2.predict(x_test)
        replace_pred = model3.predict(x_test)
        
        gscore = gms(y_test,origin_pred, average = "weighted")
        f_measure = f1_score(y_test, origin_pred, average='weighted')
        
        f_rp = f1_score(y_test, replace_pred, average='weighted')
        g_rp = gms(y_test, replace_pred, average='weighted')
        print("Final F-measure: ", f_measure)
        

        
        return f_measure,gscore, f_rp,g_rp
    
    def test(self):
        im_ratio = [10,100,200]
        sep_condition = [1,4,8]
        
        params_dict1 = {'test__max_depth': [None],
                      'test__max_features': sp_randint(1, 3),      #Number of features per bag
                      'test__min_samples_split': sp_randint(2, 100), #Min number of samples in a leaf node split
                      'test__min_samples_leaf': sp_randint(1, 100),  #Min number of samples in a leaf node
                      'test__bootstrap': [True, False],             #Sample \mathbf{x}x with/without replacement
                      'test__n_estimators' :[1,2,5,10,50,75,100,250,500,1000], #Number of trees in the forest
                      "test__n_jobs": [-1],
                      "test__class_weight": ["balanced", "balanced_subsample"]}
        
        params_dict2 = {'test__n_estimators' :[1,2,5,10,50,75,100,250,500],
                        "test__n_jobs":[-1],
                        "test__replacement":[True,False]}
        
        
        
        params_dict3 = {'test__n_estimators' :[1,2,5,10,50,75,100],
                        "test__algorithm":["SAMME", "SAMME.R"],
                        "test__replacement":[True,False]}
        
        params_dict4 = {'test__n_estimators' :[1,2,5,10,50,75,100,250,500,1000],
                        "test__n_jobs": [-1],
                        "test__replacement":[True,False],
                        "test__base_estimator":[SVC(C=1.0, cache_size=200, class_weight="balanced", coef0=0.0,
                     decision_function_shape='ovr', degree=3, gamma='auto',
                     kernel='rbf', max_iter=-1, probability=False, random_state=None,
                     shrinking=True, tol=0.001, verbose=False)],
                        "test__max_samples": sp_unif(0,1),
                        "test__max_features": sp_unif(0,1)}
        
        ensembler = [("BalancedRandomForestClassifier", BalancedRandomForestClassifier, params_dict1),
                     ("EasyEnsembleClassifier",EasyEnsembleClassifier, params_dict2),
                     ("RUSBoostClassifier",RUSBoostClassifier, params_dict3),
                     ("BalancedBaggingClassifier",BalancedBaggingClassifier, params_dict4)]
        
        # with open("Ensembler.txt", "a") as f:
        #     f.write("Strategy\timbalanced_ratio\tclass_separability\tk_J1\tu_J1\tcv_score\tcv_gms\treplace_score\treplace_gms\n")
        
        for classifier in ensembler:
            for sep in sep_condition:
                for ratio in im_ratio:
                    X_train,Y_train, X_test,Y_test = self.load_data(ratio = ratio, sep = sep)
                    
                    batch = len(X_train)

                    
                    for i in range(1):
                        
                        self.classifier = classifier[1]
                        self.params_dict = classifier[2]
                        
                        
                        self.X_train,self.Y_train,self.X_test,self.Y_test = X_train[i],Y_train[i], X_test[i],Y_test[i]
                        
                        k_J1,u_J1 = self.J1_estimate(self.X_train,self.Y_train,self.X_test,self.Y_test)
                        
                        f1_cv,g_cv,f1_rp, g_rp = self.phase2(self.X_train, self.Y_train)
                        
                        
                        with open("Ensembler.txt", "a") as f:
                            f.write("{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(classifier[0], ratio,sep,k_J1, u_J1, f1_cv, g_cv,f1_rp,g_rp))
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 14:50:19 2020

@author: ZYCBl
"""
import random
import numpy as np

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



warnings.filterwarnings("ignore")



# This code is implemented all by Yuchen Zhu. The code is the initial code file that contains almost all the code we 
# uses during the project
# This code is uploaded just for record. If interested, you can also try to see what did we do, but there's no good comments. Important
# Section of this file has been separated to smaller file. Check these for more information. 
# This code is not meant for running unless provided training and testing set properly 





class RTSCV(ClassifierMixin):
    
    def __init__(self, classifier = SVC):
        self.base = classifier
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.X_res = None
        self.Y_res = None
        
        
        

        self.model = None
        self.seperability = 0
        
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
        
        print("Known class J_1 score: ", k_J_1)
        print("Unknown class J_1 score: ", u_J_1)
        
        return k_J_1, u_J_1
        
    def J1_estimate(self,X_train,Y_train, X_test, Y_test):
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



    
    def generate_data(self,known_num = 10, unknown_num = 1, dim = 3, class_size = 100, uu_size = 300, ratio = 1,shape = 1,sep = 10):
        # Known_num  Number of Known classes
        # Unknown_num Number of Unknown classes
        # dim  dimension of data
        # class_size   average sample numbers for each known class
        # uu_size   sample numbers for each unknown class
        # ratio    the imbalanced ratio for dataset, set to 1 for balanced data
        # shape    the shape of pareto distribution, see pareto distribution
        # sep      the way to control class seperabiliy, the higher the better class separability, set to 10 as default for a medium level of separability
        
        
        
        k_cov = np.array([np.random.rand(dim,dim) for i in range(known_num)])
        k_mean = np.array([sep* np.random.rand(dim) for i in range(known_num)])
        sq = np.linspace(1, ratio ** (1/2), known_num)
        size = pareto.pdf(sq,b = shape)
        print(size)
        size_ratio = size/size.sum()

        size = np.array(size/size.sum() * class_size * known_num + 1,dtype = np.int32)
        print(max(size)/min(size))
        k_X = [multivariate_normal(k_mean[i],k_cov[i],size[i]) for i in range(known_num)]
        k_X = np.concatenate(k_X)
        k_Y = [i * np.ones((size[i],)) for i in range(known_num)]
        k_Y = np.concatenate(k_Y)
        
        
        
        x_train, x_test, y_train, y_test = train_test_split(k_X,k_Y, test_size=0.2, shuffle = True)
        
        uu_cov = np.asarray([np.random.rand(dim,dim) for i in range(unknown_num)])
        uu_mean = np.asarray([sep* np.random.rand(dim) + np.array([5,0,0]) for i in range(unknown_num)])
        uu_X = [multivariate_normal(uu_mean[i],uu_cov[i],uu_size) for i in range(unknown_num)]
        uu_X = np.concatenate(uu_X)
        uu_Y = [99 * np.ones((uu_size,)) for i in range(unknown_num)]
        uu_Y = np.concatenate(uu_Y)
        
        self.separability = self.J1_score(size_ratio,k_mean,k_cov,uu_mean, dim)
        
        
        x_test = np.concatenate((x_test, uu_X))
        y_test = np.concatenate((y_test, uu_Y))
        
        zipped = list(zip(x_test, y_test))  
        
        random.shuffle(zipped)
        x_test, y_test = zip(*zipped)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        
        self.X_train = x_train
        self.X_test = x_test
        self.Y_train = y_train
        self.Y_test = y_test
    
        print("Shape of the original training data: ", x_train.shape)
        print("Shape of the original test data: ", x_test.shape)
        
    def random_sampling(self,sample_size = 0.2):
        # Create a test sample
        # sample_size:  the percentage of sample in the test set
        
        x_test = self.X_test
        y_test = self.Y_test
        x_new_test, sample, y_new_test, sample_label = train_test_split(x_test,y_test, test_size = sample_size)
        

        return sample, sample_label, x_new_test, y_new_test
    
    def phase1(self):
        
        #Create Benchmark
        #Usual way of train classifier without RTSCV
        
        model = self.base(C=1.0, cache_size=200, class_weight="balanced", coef0=0.0,
                               decision_function_shape='ovr', degree=3, gamma='auto',
                               kernel='rbf', max_iter=-1, probability=False, random_state=None,
                               shrinking=True, tol=0.001, verbose=False)
        x_train,y_train = self.X_train, self.Y_train
        x_test,y_test = self.X_test,self.Y_test
        
        model.fit(x_train, y_train)
        origin_pred = model.predict(x_test)

        gscore = gms(y_test,origin_pred, average = "weighted")
        f_measure = f1_score(y_test, origin_pred, average='weighted')
        print("\nOriginal F-measure: ", f_measure)

        return f_measure,gscore
    def cross_validation(self, X,Y, k = 3):
        # Cross validation in RTSCV to predict Unknown classes
        # k  number of folds, doesn't affect the result a lot in a balanced setting, set to a lower value to save computational power
        
        kf = KFold(n_splits=k)
        uu_count = 0
        uu_total = []
        for train_index, test_index in kf.split(X, Y):
            xc_train, xc_test = X[train_index], X[test_index]
            yc_train, yc_test = Y[train_index], Y[test_index]
        

            cross_model = self.base(C=1.0, cache_size=200, class_weight="balanced", coef0=0.0,
                              decision_function_shape='ovr', degree=3, gamma='auto',
                              kernel='rbf', max_iter=-1, probability=False, random_state=None,
                              shrinking=True, tol=0.001, verbose=False)
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
    
        model2.fit(x_train2, y_train2)
        self.model = model2
        
        origin_pred = model2.predict(x_test)
        gscore = gms(y_test,origin_pred, average = "weighted")
        f_measure = f1_score(y_test, origin_pred, average='weighted')
        print("Final F-measure: ", f_measure)
        

        
        return f_measure,gscore
    
    
    def resample(self,resampler):
        print('Distribution before imbalancing: {}'.format(Counter(self.Y_train)))
        self.X_res,self.Y_res = resampler.fit_resample(self.X_train, self.Y_train)
        print('Resampled dataset shape %s' % Counter(self.Y_res))
        return self.phase2(self.X_res, self.Y_res)
        

    

    def save_data(self, ratio, sep, batch):
        train_list = []
        test_list = []
        
        
        for i in range(batch):
            self.generate_data(ratio = ratio, sep = sep)
            X_train = self.X_train
            Y_train = self.Y_train
            X_test = self.X_test
            Y_test = self.Y_test
            Y_train = Y_train.reshape(-1,1)
            Y_test = Y_test.reshape(-1,1)
            train = np.hstack((X_train,Y_train))
            test = np.hstack((X_test,Y_test))
            
            train_list.append(train)
            test_list.append(test)
        train_list = np.array(train_list)
        test_list = np.array(test_list)
        print(train_list.shape)
        
        
        np.save("{}-{} trainset.npy".format(ratio,sep), train_list)
        np.save("{}-{} testset.npy".format(ratio,sep), test_list)
        
    def load_data(self,ratio, sep):
        train = np.load("dataset/{}-{} trainset.npy".format(ratio,sep))
        test =  np.load("dataset/{}-{} testset.npy".format(ratio,sep))
        
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
    
    def generate_dataset(self):
        
        for ratio in [1,10,50,100]:
            for sep in [1,3,5,7,10]:
                self.save_data(ratio,sep, 1)
            

    
    
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

class plot_data(RTSCV):
    def __init__(self):
        super().__init__()
        self.k_mean = np.array([np.random.rand(6) for i in range(2)])
        self.k_cov = np.array([np.random.rand(6,6) for i in range(2)])
        self.uu_cov = np.asarray([np.random.rand(6,6) for i in range(1)])
        self.uu_mean = np.asarray([np.random.rand(6) + np.array([1,0,0,0,0,0])for i in range(1)])
        
        
    def generate_data(self,known_num = 2, unknown_num = 1, dim = 6, class_size = 1000, uu_size = 300, ratio = 1,shape = 1,sep = 10):
        # Known_num  Number of Known classes
        # Unknown_num Number of Unknown classes
        # dim  dimension of data
        # class_size   average sample numbers for each known class
        # uu_size   sample numbers for each unknown class
        # ratio    the imbalanced ratio for dataset, set to 1 for balanced data
        # shape    the shape of pareto distribution, see pareto distribution
        # sep      the way to control class seperabiliy, the higher the better class separability, set to 10 as default for a medium level of separability
        
        
        
        # k_cov = np.array([np.random.rand(dim,dim) for i in range(known_num)])
        # k_mean = np.array([sep* np.random.rand(dim) for i in range(known_num)])
        
        k_cov = self.k_cov
        k_mean = sep* self.k_mean
        
        sq = np.linspace(1, ratio ** (1/2), known_num)
        size = pareto.pdf(sq,b = shape)

        size_ratio = size/size.sum()

        size = np.array(size/size.sum() * class_size * known_num + 1,dtype = np.int32)
        
        

        k_X = [multivariate_normal(k_mean[i],k_cov[i],size[i]) for i in range(known_num)]
        k_X = np.concatenate(k_X)
        k_Y = [i * np.ones((size[i],)) for i in range(known_num)]
        k_Y = np.concatenate(k_Y)
        
        test_X = [multivariate_normal(k_mean[i],k_cov[i],size[i]) for i in range(known_num)]
        test_X = np.concatenate(test_X)
        test_Y = [i * np.ones((size[i],)) for i in range(known_num)]
        test_Y = np.concatenate(test_Y)
        # x_train, x_test, y_train, y_test = train_test_split(k_X,k_Y, test_size=0.2, shuffle = True)
        
        # uu_cov = np.asarray([np.random.rand(dim,dim) for i in range(unknown_num)])
        # uu_mean = np.asarray([sep* np.random.rand(dim) + np.array([5,0,0,0,0,0])for i in range(unknown_num)])
        uu_cov = self.uu_cov
        uu_mean = sep * self.uu_mean
        
        
        uu_X = [multivariate_normal(uu_mean[i],uu_cov[i],uu_size) for i in range(unknown_num)]
        uu_X = np.concatenate(uu_X)
        uu_Y = [99 * np.ones((uu_size,)) for i in range(unknown_num)]
        uu_Y = np.concatenate(uu_Y)
        
        self.separability = self.J1_score(size_ratio,k_mean,k_cov,uu_mean, dim)
        
        
        # x_test = np.concatenate((x_test, uu_X))
        # y_test = np.concatenate((y_test, uu_Y))
        
        # zipped = list(zip(x_test, y_test))  
        
        # random.shuffle(zipped)
        # x_test, y_test = zip(*zipped)
        # x_test = np.array(x_test)
        # y_test = np.array(y_test)
        
        # self.X_train = x_train
        # self.X_test = x_test
        # self.Y_train = y_train
        # self.Y_test = y_test
        
        
        return k_X, k_Y, uu_X,uu_Y, test_X,test_Y


    def make_data(self):
        sep = 7
        X,y,uu_x,uu_y,test_X,test_y = self.generate_data(known_num = 2, class_size = 3000, dim = 6, sep = sep, uu_size = 150)
        
        ratio_list = list(range(1,200,5)) + [200]
        for ratio in ratio_list:
            share = ceil(750/(ratio+1))
            major = share*ratio
            minor = share*1
            
            test_share = ceil(250/2)
            test_major = test_share
            test_minor = test_share
            
            distribution = {0:major,1:minor}
            distribution2 = {0:test_major, 1:test_minor}
            x_train,y_train = make_imbalance(X,y, sampling_strategy = distribution)
            x_test,y_test = make_imbalance(test_X,test_y, sampling_strategy = distribution2)
            
                
            x_test = np.concatenate((x_test, uu_x))
            y_test = np.concatenate((y_test, uu_y))
        
            zipped = list(zip(x_test, y_test))  
        
            random.shuffle(zipped)
            x_test, y_test = zip(*zipped)
            x_test = np.array(x_test)
            y_test = np.array(y_test)
            
            y_train = y_train.reshape(-1,1)
            y_test = y_test.reshape(-1,1)
            train = np.hstack((x_train,y_train))
            test = np.hstack((x_test,y_test))
            
            train_list = [train]
            test_list = [test]
            train_list = np.array(train_list)
            test_list = np.array(test_list)
            print(train_list.shape)
        
        
            np.save("dataset/{}-{} trainset.npy".format(ratio,sep), train_list)
            np.save("dataset/{}-{} testset.npy".format(ratio,sep), test_list)
    
    def section1_test(self):
        im_ratio = list(range(1,200,5)) + [200]
        sep_condition = [1,4,7,10]
        for sep in sep_condition:
            
            ####  Data Order: IR, benchmarkf1, cvf1, benchmarkgm, cvgm
            
            for ratio in im_ratio:
                X_train,Y_train, X_test,Y_test = self.load_data(ratio = ratio, sep = sep)
                batch = len(X_train)

                for i in range(batch):
                    self.X_train,self.Y_train,self.X_test,self.Y_test = X_train[i],Y_train[i], X_test[i],Y_test[i]
                        
                    f1_raw,g_raw = self.phase1()
                    
                    f1_cv,g_cv = self.phase2(self.X_train, self.Y_train)
                    
                    with open("section1_sep={}.txt".format(sep), "a") as f:
                        f.write("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(ratio,f1_raw,f1_cv,g_raw,g_cv))

    def section2_test(self):
        sep_condition = np.linspace(0.5,10,40)
        ratio_condition = [1,50,100,200]
        known_num = 5
        train_class_size = 1000
        test_class_size = 300
        
        self.k_mean = np.array([np.random.rand(6) for i in range(5)])
        self.k_cov = np.array([np.random.rand(6,6) for i in range(5)])
        self.uu_cov = np.asarray([np.random.rand(6,6) for i in range(1)])
        self.uu_mean = np.asarray([np.random.rand(6) + np.array([1,0,0,0,0,0])for i in range(1)])
        
        for sep in sep_condition:
            X,y,uu_x,uu_y,test_X,test_y = self.generate_data(known_num = known_num, class_size = 2000, dim = 6, sep = sep, uu_size = 150)
            for ratio in ratio_condition:
                k_J, u_J = self.separability
                
                
                sq = np.linspace(1, ratio ** (1/2), known_num)
                size = pareto.pdf(sq,b = 1)

                size_ratio = size/size.sum()
                
                
                balance_ratio = 1/known_num * np.ones_like(size_ratio)
                
                train_size = np.array(size_ratio * train_class_size + 1,dtype = np.int32)
                test_size = np.array(balance_ratio * test_class_size + 1,dtype = np.int32)
                
                
                order = list(range(known_num))
                random.shuffle(order)
                distribution = dict(zip(order, train_size))
                distribution2 = dict(zip(order,test_size))
                
                
                # share = ceil(750/(ratio+1))
                # major = share*ratio
                # minor = share*1
            
                # test_share = ceil(250/(ratio+1))
                # test_major = test_share * ratio
                # test_minor = test_share * 1
            
                # distribution = {0:major,1:minor}
                # distribution2 = {0:test_major, 1:test_minor}
                
                x_train,y_train = make_imbalance(X,y, sampling_strategy = distribution)
                x_test,y_test = make_imbalance(test_X,test_y, sampling_strategy = distribution2)
            
                
                x_test = np.concatenate((x_test, uu_x))
                y_test = np.concatenate((y_test, uu_y))
        
                zipped = list(zip(x_test, y_test))  
        
                random.shuffle(zipped)
                x_test, y_test = zip(*zipped)
                x_test = np.array(x_test)
                y_test = np.array(y_test)
                
                self.X_train,self.Y_train,self.X_test,self.Y_test = x_train,y_train,x_test, y_test
                
                f1_raw,g_raw = self.phase1()
                    
                f1_cv,g_cv = self.phase2(self.X_train, self.Y_train)
                    
                with open("section2_ratio={}.txt".format(ratio), "a") as f:
                    f.write("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(ratio,k_J,u_J, f1_raw,f1_cv,g_raw,g_cv))
                
                self.X_train,self.Y_train,self.X_test,self.Y_test = None,None,None,None
        

class s3_sample(sample_stategy):
        
    
    def generate_dataset(self):
        im_ratio = [10,100,200]
        sep_condition = [1,4,8]

        known_num = 10
        train_class_size = 3000
        test_class_size = 750
        
                
        self.k_mean = np.array([np.random.rand(3) for i in range(known_num)])
        self.k_cov = np.array([np.random.rand(3,3) for i in range(known_num)])
        self.uu_cov = np.asarray([np.random.rand(3,3) for i in range(known_num)])
        self.uu_mean = np.asarray([np.random.rand(3) for i in range(1)])
        
        
        for sep in sep_condition:
            X,y,uu_x,uu_y,test_X,test_y = self.generate_data(known_num = known_num, class_size = 3000, dim = 3, sep = sep, uu_size = 450)
            for ratio in im_ratio:
                
                
                sq = np.linspace(1, ratio ** (1/2), known_num)
                size = pareto.pdf(sq,b = 1)

                size_ratio = size/size.sum()
                balance_ratio = 1/known_num *np.ones_like(size_ratio)

                train_size = np.array(size_ratio * train_class_size + 1,dtype = np.int32)
                test_size = np.array(balance_ratio * test_class_size + 1,dtype = np.int32)
                
                
                order = list(range(known_num))
                random.shuffle(order)
                distribution = dict(zip(order, train_size))
                print(distribution)
                distribution2 = dict(zip(order,test_size))
                
                x_train,y_train = make_imbalance(X,y, sampling_strategy = distribution)
                x_test,y_test = make_imbalance(test_X,test_y, sampling_strategy = distribution2)
            
                
                x_test = np.concatenate((x_test, uu_x))
                y_test = np.concatenate((y_test, uu_y))
        
                zipped = list(zip(x_test, y_test))  
        
                random.shuffle(zipped)
                x_test, y_test = zip(*zipped)
                x_test = np.array(x_test)
                y_test = np.array(y_test)
                
                self.X_train,self.Y_train,self.X_test,self.Y_test = x_train,y_train,x_test, y_test
                
                self.save_data(ratio,sep,1)
                
    
    def generate_data(self,known_num = 2, unknown_num = 1, dim = 6, class_size = 1000, uu_size = 300, ratio = 1,shape = 1,sep = 10):
        # Known_num  Number of Known classes
        # Unknown_num Number of Unknown classes
        # dim  dimension of data
        # class_size   average sample numbers for each known class
        # uu_size   sample numbers for each unknown class
        # ratio    the imbalanced ratio for dataset, set to 1 for balanced data
        # shape    the shape of pareto distribution, see pareto distribution
        # sep      the way to control class seperabiliy, the higher the better class separability, set to 10 as default for a medium level of separability
        
        
        
        # k_cov = np.array([np.random.rand(dim,dim) for i in range(known_num)])
        # k_mean = np.array([sep* np.random.rand(dim) for i in range(known_num)])
        
        k_cov = self.k_cov
        k_mean = sep* self.k_mean
        
        sq = np.linspace(1, ratio ** (1/2), known_num)
        size = pareto.pdf(sq,b = shape)

        size_ratio = size/size.sum()

        size = np.array(size/size.sum() * class_size * known_num + 1,dtype = np.int32)
        
        

        k_X = [multivariate_normal(k_mean[i],k_cov[i],size[i]) for i in range(known_num)]
        k_X = np.concatenate(k_X)
        k_Y = [i * np.ones((size[i],)) for i in range(known_num)]
        k_Y = np.concatenate(k_Y)
        
        test_X = [multivariate_normal(k_mean[i],k_cov[i],size[i]) for i in range(known_num)]
        test_X = np.concatenate(test_X)
        test_Y = [i * np.ones((size[i],)) for i in range(known_num)]
        test_Y = np.concatenate(test_Y)
        # x_train, x_test, y_train, y_test = train_test_split(k_X,k_Y, test_size=0.2, shuffle = True)
        
        # uu_cov = np.asarray([np.random.rand(dim,dim) for i in range(unknown_num)])
        # uu_mean = np.asarray([sep* np.random.rand(dim) + np.array([5,0,0,0,0,0])for i in range(unknown_num)])
        uu_cov = self.uu_cov
        uu_mean = sep * self.uu_mean
        
        
        uu_X = [multivariate_normal(uu_mean[i],uu_cov[i],uu_size) for i in range(unknown_num)]
        uu_X = np.concatenate(uu_X)
        uu_Y = [99 * np.ones((uu_size,)) for i in range(unknown_num)]
        uu_Y = np.concatenate(uu_Y)
        
        self.separability = self.J1_score(size_ratio,k_mean,k_cov,uu_mean, dim)
        
        
        # x_test = np.concatenate((x_test, uu_X))
        # y_test = np.concatenate((y_test, uu_Y))
        
        # zipped = list(zip(x_test, y_test))  
        
        # random.shuffle(zipped)
        # x_test, y_test = zip(*zipped)
        # x_test = np.array(x_test)
        # y_test = np.array(y_test)
        
        # self.X_train = x_train
        # self.X_test = x_test
        # self.Y_train = y_train
        # self.Y_test = y_test
        
        
        return k_X, k_Y, uu_X,uu_Y, test_X,test_Y
    
    def save_data(self, ratio, sep, batch):
        train_list = []
        test_list = []
        
        
        for i in range(batch):
            X_train = self.X_train
            Y_train = self.Y_train
            X_test = self.X_test
            Y_test = self.Y_test
            Y_train = Y_train.reshape(-1,1)
            Y_test = Y_test.reshape(-1,1)
            train = np.hstack((X_train,Y_train))
            test = np.hstack((X_test,Y_test))
            
            train_list.append(train)
            test_list.append(test)
        train_list = np.array(train_list)
        test_list = np.array(test_list)
        print(train_list.shape)
        
        
        np.save("dataset/{}-{} trainset.npy".format(ratio,sep), train_list)
        np.save("dataset/{}-{} testset.npy".format(ratio,sep), test_list)
        
            


class OnevsrestSMBT(OneVsRestClassifier):
    
    def __init__(self,n_samples=100,
                 k_neighbors=5,
                 n_estimators=50):
        
        self.n_samples = n_samples
        self.k_neighbors = k_neighbors
        self.n_estimators = n_estimators
    
    def fit(self,X,y):
        base = ce.SMOTEBoost(n_samples = self.n_samples,k_neighbors = self.k_neighbors, n_estimators = self.n_estimators)
        self.model = OneVsRestClassifier(base)
        self.model.fit(X,y)
        
        return self
    def predict(self,X):
        
        return self.model.predict(X)
    
    def predict_proba(self,X):
        return self.model.predict_proba(X)


class OnevsrestRUS(OneVsRestClassifier):
    def __init__(self,n_samples=100,
                 with_replacement=True,
                 n_estimators=10):
        self.n_samples = n_samples
        self.with_replacement = with_replacement
        self.n_estimators = n_estimators
    
    def fit(self,X,y):
        base = ce.RUSBoost(n_samples = self.n_samples, with_replacement = self.with_replacement, n_estimators = self.n_estimators)
        self.model = OneVsRestClassifier(base)
        self.model.fit(X,y)
        
        return self
    def predict(self,X):
        
        return self.model.predict(X)
    
    def predict_proba(self,X):
        return self.model.predict_proba(X)

class OnevsrestSMBG(OneVsRestClassifier):
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
    
    def fit(self,X,y):
        base = ce.SMOTEBagging(n_estimators = self.n_estimators)
        self.model = OneVsRestClassifier(base)
        self.model.fit(X,y)
        
        return self
    def predict(self,X):
        
        return self.model.predict(X)
    
    def predict_proba(self,X):
        return self.model.predict_proba(X)

class OnevsrestUB(OneVsRestClassifier):
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
    
    def fit(self,X,y):
        base = ce.UnderBagging(n_estimators = self.n_estimators)
        self.model = OneVsRestClassifier(base)
        self.model.fit(X,y)
        
        return self
    def predict(self,X):
        
        return self.model.predict(X)
    
    def predict_proba(self,X):
        return self.model.predict_proba(X)


class OnevsrestBC(OneVsRestClassifier):
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators

    def fit(self,X,y):
        base = ce.BalanceCascade(n_estimators = self.n_estimators)
        self.model = OneVsRestClassifier(base)
        self.model.fit(X,y)
        
        return self
    def predict(self,X):
        
        return self.model.predict(X)
    
    def predict_proba(self,X):
        return self.model.predict_proba(X)

class OnevsrestSPE(OneVsRestClassifier):
    def __init__(self, n_estimators=10, k_bins=10):

        self.n_estimators_ = n_estimators
        self.k_bins_ = k_bins

    def fit(self,X,y):
        
        base = spe.SelfPacedEnsemble(n_estimators = self.n_estimators_, k_bins = self.k_bins_)

        self.model = OneVsRestClassifier(base)

        self.model.fit(X,y)
        
        return self
    def predict(self,X):
        
        return self.model.predict(X)
    
    def predict_proba(self,X):
        return self.model.predict_proba(X)
    
    
  
class OnevsrestRAMO(OneVsRestClassifier):
    def __init__(self,n_samples=100,
                 k_neighbors_1=5,
                 k_neighbors_2=5,
                 n_estimators=50):
        self.k_neighbors_1 = k_neighbors_1
        self.k_neighbors_2 = k_neighbors_2
        self.n_estimators = n_estimators
        self.n_samples = n_samples

    def fit(self,X,y):
        base = ramo.RAMOBoost(k_neighbors_1 = self.k_neighbors_1, k_neighbors_2 = self.k_neighbors_2, 
                            n_estimators = self.n_estimators,
                            n_samples = self.n_samples)
        
        self.model = OneVsRestClassifier(base)
        self.model.fit(X,y)
        
        return self
    def predict(self,X):
        
        return self.model.predict(X)
    
    def predict_proba(self,X):
        return self.model.predict_proba(X)



class s3_ensemble(ensemble):
    
    
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
        
        # params_dict1 = {'test__n_estimators' :[10,50,75,100,200],
        #                 "test__k_neighbors": sp_randint(1,20),
        #                 "test__n_samples": sp_randint(50,300)}
        
        # params_dict2 = {'test__n_estimators' :[10,50,75,100,200],
        #                 "test__n_samples": sp_randint(50,300),
        #                 "test__with_replacement":[True, False]}
        
        # params_dict3 = {'test__n_estimators' :[10,50,75,100,200]}
        
        # params_dict4 = {'test__n_estimators' :[10,50,75,100,200]}
        
        # params_dict5 = {'test__n_estimators' :[10,50,75,100,200]}
        
        # params_dict6 = {'test__n_estimators' :[10,50,75,100,200],
        #                 "test__k_bins": sp_randint(5,15)}
        
        # params_dict7 = {"test__n_estimators" :[10,50,75,100,200],
        #                 "test__n_samples": sp_randint(50,300),
        #                 "test__k_neighbors_1": sp_randint(1,20),
        #                 "test__k_neighbors_2": sp_randint(1,20)}
        
        # ensembler = [("SMOTEBoost", OnevsrestSMBT, params_dict1),
        #               ("RUSBoost",OnevsrestRUS, params_dict2),
        #               ("SMOTEBagging",OnevsrestSMBG, params_dict3),
        #               ("UnderBagging",OnevsrestUB, params_dict4),
        #               ("BalanceCascade",OnevsrestBC, params_dict5),
        #               ("SelfPacedEnsemble",OnevsrestSPE, params_dict6),
        #               ("RAMOBoost",OnevsrestRAMO, params_dict7)]
        
        
        params_dict1 = {'n_estimators' :100,
                        "k_neighbors": 5,
                        "n_samples": 150}
        
        params_dict2 = {'n_estimators' :100,
                        "n_samples": 150}
        
        params_dict3 = {'n_estimators' :100}
        
        params_dict4 = {'n_estimators' :100}
        
        params_dict5 = {'n_estimators' :100}
        
        params_dict6 = {'n_estimators' :100,
                        "k_bins": 10}
        
        params_dict7 = {"n_estimators" :100,
                        "n_samples": 150,
                        "k_neighbors_1": 5,
                        "k_neighbors_2": 5}
        
        ensembler = [ ("SelfPacedEnsemble",OnevsrestSPE, params_dict6),
                      ("RAMOBoost",OnevsrestRAMO, params_dict7)]
        
        

        
        # with open("Ensembler.txt", "a") as f:
        #     f.write("Strategy\timbalanced_ratio\tclass_separability\tk_J1\tu_J1\tcv_score\tcv_gms\treplace_score\treplace_gms\n")
        
        for classifier in ensembler:
            
            for sep in sep_condition:
                for ratio in im_ratio:
                    X_train,Y_train, X_test,Y_test = self.load_data(ratio = ratio, sep = sep)
                    
                    batch = len(X_train)

                    
                    for i in range(1):
                        
                        self.classifier = classifier[1]
                        self.kwargs = classifier[2]
                        
                        self.X_train,self.Y_train,self.X_test,self.Y_test = X_train[i],Y_train[i], X_test[i],Y_test[i]
                        
                        k_J1,u_J1 = self.J1_estimate(self.X_train,self.Y_train,self.X_test,self.Y_test)
                        
                        f1_cv,g_cv,f1_rp, g_rp = self.phase2(self.X_train, self.Y_train)
                        
                        
                        with open("s3_ensemble.txt", "a") as f:
                            f.write("{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(classifier[0], ratio,sep,k_J1, u_J1, f1_cv, g_cv,f1_rp,g_rp))
def tsne_plot():
    model = SVC(C=1.0, cache_size=200, class_weight="balanced", coef0=0.0,
                     decision_function_shape='ovr', degree=3, gamma='auto',
                     kernel='rbf', max_iter=-1, probability=False, random_state=None,
                     shrinking=True, tol=0.001, verbose=False)
    X_train,Y_train, X_test,Y_test = CV.load_data(10,10)
    X_train,Y_train, X_test,Y_test = X_train[0],Y_train[0], X_test[0],Y_test[0]
    sampler = SVMSMOTE(n_jobs = -1)
    model.fit(X_train,Y_train)


    print(Counter(Y_train))
    Y_pred = model.predict(X_test)
    print(f1_score(Y_test,Y_pred, average = "weighted"))
    X_res,Y_res = sampler.fit_resample(X_train,Y_train)
    
    model2 = SVC(C=1.0, cache_size=200, class_weight="balanced", coef0=0.0,
                     decision_function_shape='ovr', degree=3, gamma='auto',
                     kernel='rbf', max_iter=-1, probability=False, random_state=None,
                     shrinking=True, tol=0.001, verbose=False)

    model2.fit(X_res,Y_res)
    Y2_pred = model2.predict(X_test)
    print(Counter(Y_res))
    X = TSNE(n_components = 2).fit_transform(X_res)
    df = pd.DataFrame(X, columns = ["x","y"])
    sns.scatterplot(df["x"],df["y"], hue = Y_res, palette = sns.color_palette("hls", 10))
    print(f1_score(Y_test,Y2_pred, average = "macro"))

class real(sample_stategy):
    
    def load_data_letter(self, uu_num, idx):
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
    
    
    def letter_test(self):
        
        sampler = self.under_sampler + self.over_sampler

        uu = [5,11,16]
        order = range(3)

        # with open("undersampling_strategy.txt", "a") as f:
        #     f.write("Strategy\timbalanced_ratio\tclass_separability\tk_J1\tu_J1\tcv_score\timprove_score\tcv_gms\timprove_gms\n")
        

        for resampler in sampler:
            for uu_num in uu:
                f1_cv_list = []
                f1_imp_list = []
                gm_cv_list = []
                gm_imp_list = []
                for idx in order:
                    X_train,Y_train, X_test,Y_test = self.load_data_letter(uu_num = uu_num, idx = idx)
                    batch = len(X_train)
                    for i in range(batch):
                        self.X_train,self.Y_train,self.X_test,self.Y_test = X_train[i],Y_train[i], X_test[i],Y_test[i]
                        f1_cv,g_cv = self.phase2(self.X_train, self.Y_train)
                        f1_cv_list.append(f1_cv)
                        gm_cv_list.append(g_cv)
                        try:
                            f1_imp,g_imp = self.resample(resampler[1])
                            f1_imp_list.append(f1_imp)
                            gm_imp_list.append(g_imp)
                        except:
                            f1_imp = np.float64("nan")
                            g_imp = np.float64("nan")
                            
                f1_cv = np.array(f1_cv_list).mean()
                f1_imp = np.array(f1_imp_list).mean()
                g_cv = np.array(gm_cv_list).mean()
                g_imp = np.array(gm_imp_list).mean()
                       
                with open("letter_result.txt", "a") as f:
                    f.write("{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(resampler[0], uu_num,idx,f1_cv, f1_imp, g_cv, g_imp))

        
    def load_data_pen(self, uu_num, idx):
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
    def pen_test(self):
        
        sampler = self.under_sampler + self.over_sampler

        uu = [3,5,7]
        order = range(3)

        # with open("undersampling_strategy.txt", "a") as f:
        #     f.write("Strategy\timbalanced_ratio\tclass_separability\tk_J1\tu_J1\tcv_score\timprove_score\tcv_gms\timprove_gms\n")
        

        for resampler in sampler:
            for uu_num in uu:
                f1_cv_list = []
                f1_imp_list = []
                gm_cv_list = []
                gm_imp_list = []
                for idx in order:
                    X_train,Y_train, X_test,Y_test = self.load_data_pen(uu_num = uu_num, idx = idx)
                    batch = len(X_train)
                    for i in range(batch):
                        self.X_train,self.Y_train,self.X_test,self.Y_test = X_train[i],Y_train[i], X_test[i],Y_test[i]
                        f1_cv,g_cv = self.phase2(self.X_train, self.Y_train)
                        f1_cv_list.append(f1_cv)
                        gm_cv_list.append(g_cv)
                        try:
                            f1_imp,g_imp = self.resample(resampler[1])
                            f1_imp_list.append(f1_imp)
                            gm_imp_list.append(g_imp)
                        except:
                            f1_imp = np.float64("nan")
                            g_imp = np.float64("nan")
                            
                f1_cv = np.array(f1_cv_list).mean()
                f1_imp = np.array(f1_imp_list).mean()
                g_cv = np.array(gm_cv_list).mean()
                g_imp = np.array(gm_imp_list).mean()
                       
                with open("pen_result.txt", "a") as f:
                    f.write("{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(resampler[0], uu_num,idx,f1_cv, f1_imp, g_cv, g_imp))
    
    def load_data_COIL(self,uu_num, idx):
        train = np.load("COIL20/{}-{} COIL20_train.npy".format(uu_num,idx),allow_pickle = True)
        test =  np.load("COIL20/{}-{} COIL20_test.npy".format(uu_num,idx),allow_pickle = True)
        
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

    def COIL_test(self):    
        
        sampler = self.under_sampler + self.over_sampler

        uu = [6,10,13]
        order = range(3)

        # with open("undersampling_strategy.txt", "a") as f:
        #     f.write("Strategy\timbalanced_ratio\tclass_separability\tk_J1\tu_J1\tcv_score\timprove_score\tcv_gms\timprove_gms\n")
        

        for resampler in sampler:
            for uu_num in uu:
                f1_cv_list = []
                f1_imp_list = []
                gm_cv_list = []
                gm_imp_list = []
                for idx in order:
                    X_train,Y_train, X_test,Y_test = self.load_data_COIL(uu_num = uu_num, idx = idx)
                    batch = len(X_train)
                    for i in range(batch):
                        self.X_train,self.Y_train,self.X_test,self.Y_test = X_train[i],Y_train[i], X_test[i],Y_test[i]
                        f1_cv,g_cv = self.phase2(self.X_train, self.Y_train)
                        f1_cv_list.append(f1_cv)
                        gm_cv_list.append(g_cv)
                        try:
                            f1_imp,g_imp = self.resample(resampler[1])
                            f1_imp_list.append(f1_imp)
                            gm_imp_list.append(g_imp)
                        except:
                            f1_imp = np.float64("nan")
                            g_imp = np.float64("nan")
                            
                f1_cv = np.array(f1_cv_list).mean()
                f1_imp = np.array(f1_imp_list).mean()
                g_cv = np.array(gm_cv_list).mean()
                g_imp = np.array(gm_imp_list).mean()
                       
                with open("COIL_result.txt", "a") as f:
                    f.write("{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(resampler[0], uu_num,idx,f1_cv, f1_imp, g_cv, g_imp))    
    

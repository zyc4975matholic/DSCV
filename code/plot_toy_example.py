# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 14:42:44 2020

@author: ZYCBl
"""
import random
import numpy as np


from sklearn.model_selection import KFold
from numpy.random import multivariate_normal
from sklearn.svm import SVC
from imblearn.datasets import make_imbalance
from imblearn.under_sampling import *
from imblearn.over_sampling import *

# This code is implemented all by Yuchen Zhu. The code is written for toy example in paper and presentation slides.
# This code is not meant for running unless the path in the file is modified to suit the user's working directory.


mat_cov = np.array([[0.5,0],[0,0.5]])
mu1 = np.array([-2,0])
mu2 = np.array([2,0])
mu3 = np.array([0,-3])

class1 = multivariate_normal(mu1,mat_cov,40)
class2 = multivariate_normal(mu2,mat_cov,40)
class3 = multivariate_normal(mu3,mat_cov,40)

np.save("toy_example/class1_1.npy",class1)
np.save("toy_example/class2_1.npy",class2)
np.save("toy_example/class3_1.npy",class3)


class1 = np.load("toy_example/class1_1.npy")
class2 = np.load("toy_example/class2_1.npy")
class3 = np.load("toy_example/class3_1.npy")

sampler = SMOTE(n_jobs = -1)

y = np.concatenate([np.ones((len(class1),)),2 *np.ones((len(class2),))])
x = np.concatenate([class1,class2])
x_res,y_res= sampler.fit_resample(x,y)
class1 = x_res[y_res == 1]
class2 = x_res[y_res == 2]



np.save("toy_example/class1_2.npy",class1)
np.save("toy_example/class2_2.npy",class2)
np.save("toy_example/class3_2.npy",class3)







class1 = np.load("toy_example/class1_2.npy")
class2 = np.load("toy_example/class2_2.npy")
class3 = np.load("toy_example/class3_2.npy")
new_class1,new_class31 = class1[:80], class1[80:]
new_class2, new_class32 = class2[:80], class2[80:]

new_class33,final_class33 = class3[:20], class3[20:]




new_class3 = np.concatenate([new_class31,new_class32,new_class33])

np.save("toy_example/class1_3.npy",new_class1)
np.save("toy_example/class2_3.npy",new_class2)

np.save("toy_example/class3_3.npy",final_class33)
np.save("toy_example/class4_3.npy",new_class3)



X = np.concatenate([new_class1,new_class2, new_class3])
Y = np.concatenate([np.ones((len(new_class1))),2* np.ones((len(new_class2))), 3*np.ones((len(new_class3)))])

zipped = list(zip(X,Y))  
        
random.shuffle(zipped)
x_test, y_test = zip(*zipped)
X = np.array(x_test)
Y = np.array(y_test)

kf = KFold(n_splits=5)
uu_count = 0
uu_total = []
for train_index, test_index in kf.split(X, Y):
    xc_train, xc_test = X[train_index], X[test_index]
    yc_train, yc_test = Y[train_index], Y[test_index]


    cross_model = SVC(C=1.0, cache_size=200, class_weight="balanced", coef0=0.0,
                      decision_function_shape='ovr', degree=3, gamma='auto',
                      kernel='rbf', max_iter=-1, probability=False, random_state=None,
                      shrinking=True, tol=0.001, verbose=False)
    cross_model.fit(xc_train, yc_train)

# sort out the samples classified into the new class
    pred = cross_model.predict(xc_test)
    uu = xc_test[pred == 3]
    uu_count  += len(uu)
    uu_total.append(uu)

unknown = np.concatenate(uu_total)
print(len(unknown))
np.save("toy_example/class1_4.npy",new_class1)
np.save("toy_example/class2_4.npy",new_class2)
np.save("toy_example/class3_4.npy",final_class33)
np.save("toy_example/class4_4.npy", unknown)


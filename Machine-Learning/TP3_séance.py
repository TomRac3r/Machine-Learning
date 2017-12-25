# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 15:40:31 2017

@author: thomas
"""

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.lda import LDA
#from sklearn.qda import QDA
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.decomposition import PCA
from time import time
import matplotlib.pyplot as plt
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.model_selection import train_test_split


# #############################################################################
# Load data set
data = load_breast_cancer()
X = data.data
y = data.target

# #############################################################################
# split into train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

stds = StandardScaler()
stds.fit(X_train)
X_train = stds.transform(X_train)
X_test = stds.transform(X_test)

# #############################################################################
# classification

results = []
list_of_classifiers = (('SGD', SGDClassifier()),
                       ('Logistic', LogisticRegression(max_iter=1000)),
                       ('KNN', KNeighborsClassifier(3)),
                       ('SVMlin', LinearSVC()),
                       ('SVM', SVC(kernel="linear", C=0.025)),
                       ('SVM', SVC(gamma=2, C=1)),
                       ('DecisionTree', DecisionTreeClassifier(max_depth=5)),
                       ('Adaboost', AdaBoostClassifier()),
                       ('RandomForest', RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
                       ('NaiveBayes', GaussianNB()),
                      # ('LDA', LinearDiscriminantAnalysis()),
                      # ('LDA', LDA()),
                      # ('QDA', QDA()),
                       ('ASGD', SGDClassifier(average=True)),
                       ('Passive-Aggressive I', PassiveAggressiveClassifier(loss='hinge', C=1.0)),
                       ('Passive-Aggressive II', PassiveAggressiveClassifier(loss='squared_hinge', C=1.0)),
                       ('SAG', LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / X_train.shape[0])),
                       ('Perceptron', Perceptron())
                       )

for clf_name, clf in list_of_classifiers:
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    
    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    
    acc = metrics.accuracy_score(y_test, pred)
    results.append((clf_name, acc, train_time, test_time))


# plots
# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()





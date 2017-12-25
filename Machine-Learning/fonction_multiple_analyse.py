# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 20:30:00 2017

@author: thomas
"""
from sklearn.datasets import fetch_mldata
import numpy as np
from sklearn import linear_model, tree, neighbors, svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

mnist = fetch_mldata('MNIST original')


# Le dataset principal qui contient toutes les images
print (mnist.data.shape)

# Le vecteur d'annotations associé au dataset (nombre entre 0 et 9)
print (mnist.target.shape)

# Echantillonage du dataset
sample = np.random.randint(70000, size=2500)
data = mnist.data[sample]
target = mnist.target[sample]

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.8)

def knn():
    knn = neighbors.KNeighborsClassifier(n_neighbors=7)
    knn.fit(xtrain, ytrain)
    
    error = 1 - knn.score(xtest, ytest)
print('Erreur: %f' % error)


def svmm():
    svmm = svm.SVC(3)
    svmm.fit(xtrain, ytrain)
    
    error = 1 - svmm.score(xtest, ytest)
print('Erreur: %f' % error)


def treee():
    treee = tree.DecisionTreeClassifier()
    treee.fit(xtrain, ytrain)
    error = 1 - treee.score(xtest, ytest)
print('Erreur: %f' % error)


def logisticr():
    logisticr = linear_model.LogisticRegression(C=1e7)
    logisticr.fit(xtrain, ytrain)
    
    error = 1 - logisticr.score(xtest, ytest)
print('Erreur: %f' % error)


def lineard():
    lineard = LinearDiscriminantAnalysis()
    lineard.fit(xtrain, ytrain)
    
    error = 1 - lineard.score(xtest, ytest)
print('Erreur: %f' % error)


knn()
svmm()
treee()
logisticr()
lineard()

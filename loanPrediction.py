#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 12:09:39 2018

@author: sangeeth
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Reading the dataset
train = pd.read_csv('train.csv')

#Removing loan Id column which is not useful
train = train.iloc[:,1:]

#taking care of missing data
train.dropna(inplace=True)

#taking care of categorical data
train = pd.get_dummies(train, drop_first = True) 

#Making the train and test 
print(list(train))
X_test = pd.read_csv('test.csv')
X_train = train.iloc[:,:-1]
Y_train = train.iloc[:,-1]

#Visualising the features
train.plot(kind = 'box',subplots = True, layout = (4,4), sharex = False, sharey = False)
train.hist()
plt.show()

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import model_selection
from sklearn.cross_validation import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

models = []
models.append(("LogisticRegression", LogisticRegression()))
models.append(("DecisionTreeClassifier", DecisionTreeClassifier()))
models.append(("KNeighborsClassifier", KNeighborsClassifier()))
models.append(("LinearDiscriminantAnalysis", LinearDiscriminantAnalysis()))
models.append(("GaussianNB", GaussianNB()))
models.append(("SVM", SVC()))

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    result = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
    #print("kfold iterations of " + name + " is " + str(result))
    std    = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy').std()
    print("score for the model " + str(name) + " is " + str(np.round(result.mean()*100,1)) + "% standard deviation is " + str(std) )
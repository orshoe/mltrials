#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
#from sklearn.decomposition import PCA
#plt.scatter(titanicTrain['Converted'],titanicTrain['predominantCreditScore'])

titanicTrain = read_csv("trainUpd2.csv")
#titanicTest = read_csv("testUpd2.csv")
#print(titanicTrain.info())
#print(titanicTrain.describe())

trainingdata = pd.DataFrame(titanicTrain,columns = ['Pclass','SibSp','Parch','SexUpd','Rels','EmbarkedN','CabinN','AgeN','Fare'])
#testdata = pd.DataFrame(titanicTest,columns = ['Pclass','SibSp','Parch','SexUpd','Rels','EmbarkedN','CabinN','AgeN','Fare'])

print("training data: ")
print(trainingdata.shape)

#trainingdata.hist(bins=10, figsize=(5,5))
#print("testing data: ")
#print(testdata.shape)

#testdata.hist(bins=10, figsize=(5,5))

trainingtarget = pd.DataFrame(titanicTrain, columns = ['Survived'])
print("training target: ")
print(trainingtarget.shape)
#testtarget = pd.DataFrame(titanicTest, columns = ['Survived'])
#print("testing target: ")
#print(testtarget.shape)


traindata, testdata, traintarget, testtarget = train_test_split(trainingdata, trainingtarget, test_size=0.3)
print(traindata.shape) 
print(testdata.shape)
print(traintarget.shape)
print(testtarget.shape)


knearestn = neighbors.KNeighborsClassifier(n_neighbors=5)
knearestn.fit(traindata,traintarget.values.ravel())
print("\nK Nearest Neighbors: ")
#print(knearestn.predict([[0,3,2,1,2,700]]))
print("K Nearest Neighbors Score: ")
print(knearestn.score(testdata, testtarget.values.ravel()))

logreg = linear_model.LogisticRegression()
logreg.fit(traindata,traintarget.values.ravel())
print("\nLogistic Regression: ")
#print(logreg.predict([[0,3,2,1,2,700]]))
print("Logistic Regression Score: ")
print(logreg.score(testdata, testtarget.values.ravel()))
"""
lreg = linear_model.LinearRegression(normalize=True)
lreg.fit(traindata,traintarget.values.ravel())
print("\nLinear Regression: ")
print(lreg.predict([[0,3,2,1,2,700]]))
print("Linear Regression Score: ")
print(lreg.score(testdata, testtarget.values.ravel()))

ridge = linear_model.Ridge(normalize=True)
ridge.fit(traindata,traintarget.values.ravel())
print("\nRidge Regression: ")
print(ridge.predict([[0,3,2,1,2,700]]))
print("Ridge Regression Score: ")
print(ridge.score(testdata, testtarget.values.ravel()))
"""
treemodelE = tree.DecisionTreeClassifier(criterion='entropy')
treemodelE.fit(traindata,traintarget.values.ravel())
print("\nDecisionTree Classifier (entropy): ")
#print(treemodelE.predict([[0,3,2,1,2,700]]))
print("Decision Tree Score: ")
print(treemodelE.score(testdata, testtarget.values.ravel()))

treemodelG = tree.DecisionTreeClassifier(criterion='gini')
treemodelG.fit(traindata,traintarget.values.ravel())
print("\nDecisionTree Classifier (gini): ")
#print(treemodelG.predict([[0,3,2,1,2,700]]))
print("Decision Tree Score: ")
print(treemodelG.score(testdata, testtarget.values.ravel()))

randomforest = RandomForestClassifier(n_estimators=1000)
randomforest.fit(traindata,traintarget.values.ravel())
print("\nRandom Forest Classifier: ")
#print(randomforest.predict([[0,3,2,1,2,700]]))
print("Random Forest Score: ")
print(randomforest.score(testdata, testtarget.values.ravel()))

gboost = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
gboost.fit(traindata,traintarget.values.ravel())
print("\nGradientBoostingClassifier: ")
#print(gboost.predict([[0,3,2,1,2,700]]))
print("GradientBoostingClassifier Score: ")
print(gboost.score(testdata, testtarget.values.ravel()))

nBayes = GaussianNB()
nBayes.fit(traindata,traintarget.values.ravel())
print("\nGaussian Naive Bayes: ")
#print(nBayes.predict([[0,3,2,1,2,700]]))
print("Naive Bayes Score: ")
print(nBayes.score(testdata, testtarget.values.ravel()))

supportVectorL = svm.SVC(C=100)
supportVectorL.fit(traindata,traintarget.values.ravel())
print("\nSupport Vector Linear: ")
#print(supportVectorL.predict([[0,3,2,1,2,700]]))
print("Support Vector Linear Score: ")
print(supportVectorL.score(testdata, testtarget.values.ravel()))

"""
from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3, random_state=0)
k_means.fit(traindata,traintarget.values.ravel())
print("\nKMeans: ")
print(k_means.predict([[0,3,2,1,2,700]]))
print("KMeans Score: ")
print(k_means.score(testdata, testtarget.values.ravel()))
"""
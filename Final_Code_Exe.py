#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 00:08:29 2020

@author: Shirish
"""

"""
Created on Sat Feb 22 14:37:53 2020

@author: Shirish
"""

import pandas as pd
import numpy as np

#Pre-Processing the data

df = pd.read_csv('dataset_abacus_final.csv')
df.fillna(df.mean(), inplace=True)
df = df.drop("Record number",1)

#Correlation
import seaborn as sn
corr=df.corr() 
sn.heatmap(corr, annot=True)

#Handling Categorical Values
df1=pd.get_dummies(df['Education'])
df = pd.concat([df1,df],axis=1)
df.drop('Education',axis=1,inplace=True)

df1=pd.get_dummies(df['Marital_status'])
df = pd.concat([df1,df],axis=1)
df.drop('Marital_status',axis=1,inplace=True)

df1=pd.get_dummies(df['Gendre'])
df = pd.concat([df1,df],axis=1)
df.drop('Gendre',axis=1,inplace=True)

df1=pd.get_dummies(df['Head of household'])
df = pd.concat([df1,df],axis=1)
df.drop('Head of household',axis=1,inplace=True)

df1=pd.get_dummies(df['Housing  type'])
df = pd.concat([df1,df],axis=1)
df.drop('Housing  type',axis=1,inplace=True)

X = df.drop("Bankruptcy on file",1)
y = df["Bankruptcy on file"]


#Label Encoding
from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
y = le.fit_transform(y)

   
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X = sc_x.fit_transform(df)

#Visualization
import matplotlib.pyplot as plt
plt.plot(X,y,'o')
plt.show()

#Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3)


#Importing Metrics
from sklearn.metrics import accuracy_score


#RandomForest
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier()
rf=random_forest.fit(X_train, y_train)
pre=rf.predict(X_test)
accuracy_score(y_test,pre)

#PrecisionAndRecall
from sklearn import metrics
print(metrics.classification_report(y_test, pre))
print(metrics.confusion_matrix(y_test, pre))
plt.show()


#DT
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
dt=decisiontree.fit(X_train, y_train)
pre=dt.predict(X_test)
accuracy_score(y_test,pre)

#PrecisionAndRecall
from sklearn import metrics
print(metrics.classification_report(y_test, pre))
print(metrics.confusion_matrix(y_test, pre))
plt.show()

#SVM
from sklearn.svm import SVC
    	
svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)    
s=svm.fit(X_train,y_train)    
pre=s.predict(X_test)    
accuracy_score(y_test,pre)

#PrecisionAndRecall
from sklearn import metrics
print(metrics.classification_report(y_test, pre))
print(metrics.confusion_matrix(y_test, pre))
plt.show()


#KNN
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()
kn=knn.fit(X_train,y_train)
pre=kn.predict(X_test)
accuracy_score(y_test,pre)


#PrecisionAndRecall
from sklearn import metrics
print(metrics.classification_report(y_test, pre))
print(metrics.confusion_matrix(y_test, pre))
plt.show()

#NaiveB
from sklearn.naive_bayes import GaussianNB

nbg=GaussianNB()
nb=nbg.fit(X_train,y_train)
pre = nb.predict(X_test)
accuracy_score(y_test,pre)

#PrecisionAndRecall
from sklearn import metrics
print(metrics.classification_report(y_test, pre))
print(metrics.confusion_matrix(y_test, pre))
plt.show()

#Log
from sklearn.linear_model import LogisticRegression

logReg=LogisticRegression()
lr=logReg.fit(X_train,y_train)
pre = lr.predict(X_test)
accuracy_score(y_test,pre)

#PrecisionAndRecall
from sklearn import metrics
print(metrics.classification_report(y_test, pre))
print(metrics.confusion_matrix(y_test, pre))
plt.show()









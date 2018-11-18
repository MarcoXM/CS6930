#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 22:00:26 2018
@author: marcowang
"""

import numpy as np
import pandas as pd 
import heapq
import matplotlib.pyplot as plt

#Import Trainning set and Test Set.
df1 = pd.read_csv('spam_train.csv')
df1.head()

#training set and test set 
X_train = df1.drop('class',axis = 1 ).values
y_train = df1['class'].values.reshape(-1,1)
Class = df1['class']


df2 = pd.read_csv('spam_test.csv')
df2.head()
X_test = df2.drop('Label',axis = 1 ).drop(' ID',axis = 1).values
y_test = df2['Label'].values.reshape(-1,1)
Label = list(df2['Label'])



#define a function to get accuracy
def score(list1,list2):
    
    if len(list1) == len(list2):
        
        return (list(list1 - np.array(list2)).count(0))/len(list1)  
    else:
        print('list1 & list2 should have same type' )


#testing KNN in 1(a)
#get a dataset of distances
k = [ 1, 5, 11, 21, 41, 61, 81, 101, 201, 401]

Matrixdistance =[]
y_pred = []
n = 0
for j in list(range(0,2301)):
    m = 0
    listdistance = []
    outcome = []
    
    for u in list(range(0,2300)):
        distance = (((X_test[n]-X_train[m])**2).sum())**0.5#data = data.drop(data['0'],axis = 1)
        listdistance.append(distance)
        m += 1
    Matrixdistance.append(listdistance)
    n += 1
Database = pd.DataFrame(Matrixdistance)    

Database.to_csv('Database.csv',index = False)
data = pd.read_csv('Database.csv')


you = []
for i in k:
    elector4all = []
    j = 0
    v = 0
    outcome = []
    y_pred = []
    for z in list(range(0,2301)):
        list_distance = data.iloc[j]
        target_distance = heapq.nsmallest(i,list_distance)
        min_num_index_list = list(map(list(list_distance).index, target_distance))
        j += 1
        elector4all.append(min_num_index_list)
    
    for d in elector4all:
        
        x = Class.iloc[d]
        
        oc = np.array(x)
        
        if oc.sum()*2 > i:
            y = int(1)
        
        else:
            y = int(0)
        
        v += 1
        y_pred.append(y)
    test_accuracy = score(y_pred,Label)
    you.append(test_accuracy)
print('the first accuray')
print(you)

   
plt.figure(figsize = (10,7.2))  
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(k, you, label = 'Testing Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylim((0.7,1))
plt.ylabel('Accuracy')      
        
       
 #1(b)

#standarzation 
X_train1 = df1.drop('class',axis = 1 )
X_test1 = df2.drop('Label',axis = 1 ).drop(' ID',axis = 1)
X_z = np.array((X_train1 - X_train1.mean())/X_train1.std())
X_ztest = np.array((X_test1 - X_train1.mean())/X_train1.std())


Matrixdistance_Z =[]
y_pred_Z = []
n = 0
for j in list(range(0,2301)):
    m = 0
    listdistance = []
    outcome = []
    
    for u in list(range(0,2300)):
        distance = (((X_ztest[n]-X_z[m])**2).sum())**0.5#data = data.drop(data['0'],axis = 1)
        listdistance.append(distance)
        m += 1
    Matrixdistance_Z.append(listdistance)
    n += 1
Database_Z = pd.DataFrame(Matrixdistance_Z)    

Database_Z.to_csv('Database_Z.csv',index = False)
data_Z = pd.read_csv('Database_Z.csv')


List_ac_Z = []
y_set_Z = []
for i in k:
    elector4all_Z = []
    j = 0
    v = 0
    outcome = []
    y_pred = []
    for z in list(range(0,2301)):
        list_distance = data_Z.iloc[j]
        target_distance = heapq.nsmallest(i,list_distance)
        min_num_index_list = list(map(list(list_distance).index, target_distance))
        j += 1
        elector4all_Z.append(min_num_index_list)
    
    for d in elector4all_Z:
        
        x = Class.iloc[d]
        
        oc = np.array(x)
        
        if oc.sum()*2 > i:
            y = int(1)
        
        else:
            y = int(0)
        
        v += 1
        y_pred.append(y)
    y_set_Z.append(y_pred)    
    test_accuracy = score(y_pred,Label)
    List_ac_Z.append(test_accuracy)
print('the accuracy after Z_score')
print(List_ac_Z)

   
plt.figure(figsize = (10,7.2))
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(k, List_ac_Z, label = 'Testing Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors of Z_Score')
plt.ylim((0.7,1))
plt.ylabel('Accuracy')   
plt.show()

# Dataframe of predct outcome
predict = np.array(y_set_Z).T
predict = pd.DataFrame(predict)
predict_50 = predict[0:50]
columns = ['k1', 'k5', 'k11', 'k21', 'k41', 'k61', 'k81', 'k101', 'k201', 'k401']
predict_50.columns = columns
predict_50 = predict_50.replace(1,'spam',)
predict_50 = predict_50.replace(0,'no',)
predict_50.index=list(df2[' ID'][:50])
print(predict_50)

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 23:44:54 2018

@author: 90786
"""
'''Import package we need'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

'''Adding x0 to dataframe!!'''
x1000 = pd.DataFrame([1] * 1000)
x100 = pd.DataFrame([1] * 100)

'''Training & test set'''
train1 = pd.read_csv('D:/Data Mining/HW1/train-1000-100.csv')
train2 = pd.read_csv('D:/Data Mining/HW1/train-100-100.csv')
train3 = pd.read_csv('D:/Data Mining/HW1/train-100-10.csv')               
test1 = pd.read_csv('D:/Data Mining/HW1/test-1000-100.csv')
test2 = pd.read_csv('D:/Data Mining/HW1/test-100-100.csv')
test3 = pd.read_csv('D:/Data Mining/HW1/test-100-10.csv')

#add X0 into the dataframe
train1 = pd.concat([x1000,train1],axis = 1)
train2 = pd.concat([x100,train2],axis = 1)
train3 = pd.concat([x100,train3],axis = 1)
test1 = pd.concat([x1000,test1],axis = 1)
test2 = pd.concat([x1000,test2],axis = 1)
test3 = pd.concat([x1000,test3],axis = 1)

'''train-50(1000)-100.csv, 
train-100(1000)-100.csv,
 train-150(1000)-100.csv'''
df1 = train1.iloc[0:50,:]
df2 = train1.iloc[0:100,:]
df3 = train1.iloc[0:150,:]
df1.to_csv('train-50(1000)-100.csv')
df2.to_csv('train-100(1000)-100.csv')
df3.to_csv('train-150(1000)-100.csv')

'''Question 1-a: train 1000-100 & test 1000-100'''
Xtrain1 = train1.drop('y', axis = 1).values
ytrain1 = train1['y'].values.reshape(-1,1)
Xtest1 = test1.drop('y', axis = 1).values
ytest1 = test1['y'].values.reshape(-1,1)
λ = np.array(range(0,151))
w1 = []
MSE_train1 = []
MSE_test1 = []
n = 0
for u in λ:
    if n < 151:
        w= np.dot(np.linalg.inv(Xtrain1.T.dot(Xtrain1) + λ[n]*np.eye(101)),Xtrain1.T).dot(ytrain1)
        Mr = ((Xtrain1.dot(w) - ytrain1)**2).mean()
        MSE_train1.append(Mr)
        Me = ((Xtest1.dot(w) - ytest1)**2).mean()
        MSE_test1.append(Me)
        w1.append(w)
        n += 1

print('MSE_train 1000-100 is :{}'.format(min(MSE_train1)))
print('MSE_test 1000-100 is :{}'.format(min(MSE_test1)))
print('the best λ for train 1000-100 is :{}'.format(MSE_train1.index(min(MSE_train1))))
print('the best λ for test 1000-100 is :{}'.format(MSE_test1.index(min(MSE_test1))))
plt.figure(1)
plt.plot(λ,MSE_train1, color = 'r')
plt.plot(λ,MSE_test1, color = 'b')
plt.title('1000-100 train & 1000-100 test')
plt.xlabel('Lambda')
plt.ylabel('MSE 1000-100')
plt.legend()
plt.show()


'''Question 1-a: train 100-100 & test 100-100'''
Xtrain2 = train2.drop('y', axis = 1).values
ytrain2 = train2['y'].values.reshape(-1,1)
Xtest2 = test2.drop('y', axis = 1).values
ytest2 = test2['y'].values.reshape(-1,1)
λ = np.array(range(0,151))
w2 = []
MSE_train2 = []
MSE_test2 = []
n = 0
for u in λ:
    if n < 151:
        w = np.dot(np.linalg.inv(Xtrain2.T.dot(Xtrain2) + λ[n]*np.eye(101)), Xtrain2.T).dot(ytrain2)
        Mr = ((Xtrain2.dot(w) - ytrain2)**2).mean()
        MSE_train2.append(Mr)
        Me = ((Xtest2.dot(w) - ytest2)**2).mean()
        MSE_test2.append(Me)
        w2.append(w)
        n += 1

print('MSE_train2 is :{}'.format(min(MSE_train2)))
print('MSE_test2 is :{}'.format(min(MSE_test2)))
print('the best λ for train 100-100 is :{}'.format(MSE_train2.index(min(MSE_train2))))
print('the best λ for test 100-100 is :{}'.format(MSE_test2.index(min(MSE_test2))))
plt.figure(2)
plt.plot(λ,MSE_train2, color = 'r',label = 'Training')
plt.plot(λ,MSE_test2, color = 'b',label = 'Testing')
plt.title(' 100-100 train & 100-100 test')
plt.xlabel('Lambda')
plt.ylabel('MSE 100-100')
plt.legend()
plt.show()



#3
'''Question 1-a: train 100-10 & test 100-10'''
Xtrain3 = train3.drop('y', axis = 1).values
ytrain3 = train3['y'].values.reshape(-1,1)
Xtest3 = test3.drop('y', axis = 1).values
ytest3 = test3['y'].values.reshape(-1,1)
λ = np.array(range(0,151))
w3 = []
MSE_train3 = []
MSE_test3 = []
n = 0
for u in λ:
    if n < 151:
        w = np.dot(np.linalg.inv(Xtrain3.T.dot(Xtrain3) + λ[n]*np.eye(11)),Xtrain3.T).dot(ytrain3)
        Mr = ((Xtrain3.dot(w) - ytrain3)**2).mean()
        MSE_train3.append(Mr)
        Me = ((Xtest3.dot(w) - ytest3)**2).mean()
        MSE_test3.append(Me)
        w3.append(w)
        n += 1

print('MSE_train3 is :{}'.format(min(MSE_train3)))
print('MSE_test3 is :{}'.format(min(MSE_test3)))
print('the best λ for train 100-10 is :{}'.format(MSE_train3.index(min(MSE_train3))))
print('the best λ for test 100-10 is :{}'.format(MSE_test3.index(min(MSE_test3))))
plt.figure(3)
plt.plot(λ,MSE_train3, color = 'r',label = 'Training')
plt.plot(λ,MSE_test3, color = 'b',label = 'Testing')
plt.title('First 100-10 train & 100-10 test')
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.legend()
plt.show()

print(' Summary of 1-a ')
print('the best λ for train 1000-100 is :{}'.format(MSE_train1.index(min(MSE_train1))))
print('Least MSE of 1000-100 is :{}'.format(min(MSE_train1)))
print('the best λ for test 1000-100 is :{}'.format(MSE_test1.index(min(MSE_test1))))
print('Least MSE of 1000-100 is :{}'.format(min(MSE_test1)))

print('the best λ for train 100-100 is :{}'.format(MSE_train2.index(min(MSE_train2))))
print('Least MSE of is :{}'.format(min(MSE_train2)))
print('the best λ for test 100-100 is :{}'.format(MSE_test2.index(min(MSE_test2))))
print('Least MSE of is :{}'.format(min(MSE_test2)))

print('the best λ for train 100-10is :{}'.format(MSE_train3.index(min(MSE_train3))))
print('Least MSE of is :{}'.format(min(MSE_train3)))
print('the best λ for test 100-10 is :{}'.format(MSE_test3.index(min(MSE_test3))))
print('Least MS of is :{}'.format(min(MSE_test3)))

print('--------------------------------------------------------------------------------')

'''Question 1-b: train 50(1000-100) & test 1000-100'''
X1 = df1.drop('y', axis = 1).values
y1 = df1['y'].values.reshape(-1,1)
Xt = test1.drop('y', axis = 1).values
yt = test1['y'].values.reshape(-1,1)
λ = np.array(range(0,151))
w4 = []
MSE_df1 = []
MSE_test4 = []
n = 0
for u in λ:
    if n < 151:
        w = np.dot(np.linalg.inv(X1.T.dot(X1) + λ[n]*np.eye(101)),X1.T).dot(y1)
        Mr = ((X1.dot(w) - y1)**2).mean()
        MSE_df1.append(Mr)
        Me = ((Xt.dot(w) - yt)**2).mean()
        MSE_test4.append(Me)
        w4.append(w)
        n += 1

print('the best λ for train 50(1000-100) is :{}'.format(MSE_df1.index(min(MSE_df1))))
print('MSE of train 50(1000-100) is :{}'.format(min(MSE_df1)))
print('the best λ for test 1000-100 is :{}'.format(MSE_test4.index(min(MSE_test4))))
print('MSE of test 1000-100 is :{}'.format(min(MSE_test4)))


plt.figure(4,figsize = (10,6))
plt.plot(λ,MSE_df1, color = 'r',label = 'Training')
plt.plot(λ,MSE_test4, color = 'b',label = 'Testing')
plt.title(' 50(1000-100) train & 1000-100 test')
plt.xlabel('Lambda')
plt.ylabel('MES')
plt.legend()
plt.show()

'''Question 1-b: train 100(1000-100) & test 1000-100'''
X2 = df2.drop('y', axis = 1).values
y2 = df2['y'].values.reshape(-1,1)
Xt = test1.drop('y', axis = 1).values
yt = test1['y'].values.reshape(-1,1)
λ = np.array(range(0,151))
w5 = []
MSE_df2 = []
MSE_test5 = []
n = 0
for u in λ:
    if n < 151:
        w = np.dot(np.linalg.inv(X2.T.dot(X2) + λ[n]*np.eye(101)),X2.T).dot(y2)
        Mr = ((X2.dot(w) - y2)**2).mean()
        MSE_df2.append(Mr)
        Me = ((Xt.dot(w) - yt)**2).mean()
        MSE_test5.append(Me)
        w5.append(w)
        n += 1

print('the best λ for 100(1000-100) train is :{}'.format(MSE_df2.index(min(MSE_df2))))
print('MSE of is 100(1000-100) train :{}'.format(min(MSE_df2)))
print('the best λ for 1000-100 test is :{}'.format(MSE_test5.index(min(MSE_test5))))
print('MSE of 1000-100 test is :{}'.format(min(MSE_test5)))

plt.figure(5,figsize = (10,6))
plt.plot(λ,MSE_df2, color = 'r',label = 'Training')
plt.plot(λ,MSE_test5, color = 'b',label = 'Testing')
plt.title('100(1000-100) train & 1000-100 test')
plt.xlabel('Lambda')
plt.ylabel('MES')
plt.legend()
plt.show()


'''Question 1-b: train 150（1000-100) & test 1000-100'''
X3 = df3.drop('y', axis = 1).values
y3 = df3['y'].values.reshape(-1,1)
Xt = test1.drop('y', axis = 1).values
yt = test1['y'].values.reshape(-1,1)
λ = np.array(range(0,151))
w6 = []
MSE_df3 = []
MSE_test6 = []
n = 0
for u in λ:
    if n < 151:
        w = np.dot(np.linalg.inv(X3.T.dot(X3) + λ[n]*np.eye(101)),X3.T).dot(y3)
        Mr = ((X3.dot(w) - y3)**2).mean()
        MSE_df3.append(Mr)
        Me = ((Xt.dot(w) - yt)**2).mean()
        MSE_test6.append(Me)
        w6.append(w)
        n += 1

print('the best λ for 150(1000-100) train is :{}'.format(MSE_df3.index(min(MSE_df3))))
print('MSE of 150(1000-100) train is :{}'.format(min(MSE_df3)))
print('the best λ for 1000-100 test is :{}'.format(MSE_test6.index(min(MSE_test6))))
print('MSEof 1000-100 test is :{}'.format(min(MSE_test6)))


plt.figure(6,figsize = (10,6))
plt.plot(λ,MSE_df3, color = 'r',label = 'Training')
plt.plot(λ,MSE_test6, color = 'b',label = 'Testing')
plt.title(' 150(1000-100) train & 100-100 test')
plt.xlabel('Lambda')
plt.ylabel('MSE 150(1000-100)')
plt.legend()
plt.show()

'''
#2-------------------------------------------------------------
Using CV technique, what is the best choice of  value and the corresponding test
set MSE for each of the six datasets?
'''
'''CV 1000-100 train set'''
print('--------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------')

kf = KFold(n_splits = 10)
kf.get_n_splits(train1)
train_x1 = []
test_x1 = []
train_y1 = []
test_y1 = []
for train_index, test_index in kf.split(train1):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_2train1 = train1.drop('y', axis = 1).values
    y_2train1 = train1['y'].values.reshape(-1,1)
    X_train, X_test = X_2train1[train_index], X_2train1[test_index]
    y_train, y_test = y_2train1[train_index], y_2train1[test_index]
    train_x1.append(X_train)
    test_x1.append(X_test)
    train_y1.append(y_train)
    test_y1.append(y_test)

λ = np.array(range(0,151))
R= np.array(range(0,10))
w21 = []
MSE_train21 = []
MSE_test21 = []
n = 0
for u in λ:
    if n < 151:
        m = 0
        u = λ[n]
        w = []
        Mr = []
        Me = []
        for m in R:
            if m < 10 :
                wm = np.dot(np.linalg.inv(train_x1[m].T.dot(train_x1[m]) + u*np.eye(101)), train_x1[m].T).dot(train_y1[m])
                w.append(wm)
                Mrm = ((train_x1[m].dot(wm) - train_y1[m])**2).mean()
                Mr.append(Mrm)
                Mem = ((test_x1[m].dot(wm) - test_y1[m])**2).mean()
                Me.append(Mem)
                m +=1
                
        MSE_train21.append(np.average(Mr))
        MSE_test21.append(np.average(Me))
        w21.append(np.average(w))
        n += 1

print('MSE_train is :{}'.format(min(MSE_train21)))
print('MSE_test is :{}'.format(min(MSE_test21)))
print('the best λ for training in 1000-100 train set is :{}'.format(MSE_train21.index(min(MSE_train21))))
print('the best λ for testing in 1000-100 train set is :{}'.format(MSE_test21.index(min(MSE_test21))))
plt.figure(1)
plt.plot(λ,MSE_train21, color = 'r', label = 'Training')
plt.plot(λ,MSE_test21, color = 'b', label = 'Testing')
plt.title('CV 1000-100 train set')
plt.xlabel('Lambda')
plt.ylabel('MES')
plt.legend()
plt.show()


'''CV 100-100 train set'''

kf = KFold(n_splits = 10)
kf.get_n_splits(train1)
train_x2 = []
test_x2 = []
train_y2 = []
test_y2 = []
for train_index, test_index in kf.split(train2):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_2train2 = train2.drop('y', axis = 1).values
    y_2train2 = train2['y'].values.reshape(-1,1)
    X_train, X_test = X_2train2[train_index], X_2train2[test_index]
    y_train, y_test = y_2train2[train_index], y_2train2[test_index]
    train_x2.append(X_train)
    test_x2.append(X_test)
    train_y2.append(y_train)
    test_y2.append(y_test)

λ = np.array(range(0,151))
R= np.array(range(0,10))
w22 = []
MSE_train22 = []
MSE_test22 = []
n = 0
for u in λ:
    if n < 151:
        m = 0
        u = λ[n]
        w = []
        Mr = []
        Me = []
        for m in R:
            if m < 10 :
                wm = np.dot(np.linalg.inv(train_x2[m].T.dot(train_x2[m]) + u*np.eye(101)), train_x2[m].T).dot(train_y2[m])
                w.append(wm)
                Mrm = ((train_x2[m].dot(wm) - train_y2[m])**2).mean()
                Mr.append(Mrm)
                Mem = ((test_x2[m].dot(wm) - test_y2[m])**2).mean()
                Me.append(Mem)
                m +=1
                
        MSE_train22.append(np.average(Mr))
        MSE_test22.append(np.average(Me))
        w22.append(np.average(w))
        n += 1

print('MSE_train is :{}'.format(min(MSE_train22)))
print('MSE_test is :{}'.format(min(MSE_test22)))
print('the best λ for training in 100-100 train set is :{}'.format(MSE_train22.index(min(MSE_train22))))
print('the best λ for testing in 100-100 train set is :{}'.format(MSE_test22.index(min(MSE_test22))))
plt.figure(2)
plt.plot(λ,MSE_train22, color = 'r', label = 'Training')
plt.plot(λ,MSE_test22, color = 'b', label = 'Testing')
plt.title('CV 100-100 train set')
plt.xlabel('Lambda')
plt.ylabel('MES')
plt.legend()
plt.show()


'''CV 100-10 train set'''

kf = KFold(n_splits = 10)
kf.get_n_splits(train1)
train_x3 = []
test_x3 = []
train_y3 = []
test_y3 = []
for train_index, test_index in kf.split(train3):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_2train3 = train3.drop('y', axis = 1).values
    y_2train3 = train3['y'].values.reshape(-1,1)
    X_train, X_test = X_2train3[train_index], X_2train3[test_index]
    y_train, y_test = y_2train3[train_index], y_2train3[test_index]
    train_x3.append(X_train)
    test_x3.append(X_test)
    train_y3.append(y_train)
    test_y3.append(y_test)

λ = np.array(range(0,151))
R= np.array(range(0,10))
w23 = []
MSE_train23 = []
MSE_test23 = []
n = 0
for u in λ:
    if n < 151:
        m = 0
        u = λ[n]
        w = []
        Mr = []
        Me = []
        for m in R:
            if m < 10 :
                wm = np.dot(np.linalg.inv(train_x3[m].T.dot(train_x3[m]) + u*np.eye(11)), train_x3[m].T).dot(train_y3[m])
                w.append(wm)
                Mrm = ((train_x3[m].dot(wm) - train_y3[m])**2).mean()
                Mr.append(Mrm)
                Mem = ((test_x3[m].dot(wm) - test_y3[m])**2).mean()
                Me.append(Mem)
                m +=1
                
        MSE_train23.append(np.average(Mr))
        MSE_test23.append(np.average(Me))
        w23.append(np.average(w))
        n += 1

print('MSE_train is :{}'.format(min(MSE_train23)))
print('MSE_test is :{}'.format(min(MSE_test23)))
print('the best λ for training in 100-10 train setis :{}'.format(MSE_train23.index(min(MSE_train23))))
print('the best λ for testing in 100-10 train setis :{}'.format(MSE_test23.index(min(MSE_test23))))
plt.figure(3)
plt.plot(λ,MSE_train23, color = 'r', label = 'Training')
plt.plot(λ,MSE_test23, color = 'b', label = 'Testing')
plt.title('CV 100-10 train set')
plt.xlabel('Lambda')
plt.ylabel('MES')
plt.legend()
plt.show()


'''CV 50(1000-100) test set'''

kf = KFold(n_splits = 10)
kf.get_n_splits(train1)
train_x4 = []
test_x4 = []
train_y4 = []
test_y4 = []
for train_index, test_index in kf.split(df1):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_2train4 = df1.drop('y', axis = 1).values
    y_2train4 = df1['y'].values.reshape(-1,1)
    X_train, X_test = X_2train4[train_index], X_2train4[test_index]
    y_train, y_test = y_2train4[train_index], y_2train4[test_index]
    train_x4.append(X_train)
    test_x4.append(X_test)
    train_y4.append(y_train)
    test_y4.append(y_test)

λ = np.array(range(0,151))
R= np.array(range(0,10))
w24 = []
MSE_train24 = []
MSE_test24 = []
n = 0
for u in λ:
    if n < 151:
        m = 0
        u = λ[n]
        w = []
        Mr = []
        Me = []
        for m in R:
            if m < 10 :
                wm = np.dot(np.linalg.inv(train_x4[m].T.dot(train_x4[m]) + u*np.eye(101)), train_x4[m].T).dot(train_y4[m])
                w.append(wm)
                Mrm = ((train_x4[m].dot(wm) - train_y4[m])**2).mean()
                Mr.append(Mrm)
                Mem = ((test_x4[m].dot(wm) - test_y4[m])**2).mean()
                Me.append(Mem)
                m +=1
                
        MSE_train24.append(np.average(Mr))
        MSE_test24.append(np.average(Me))
        w24.append(np.average(w))
        n += 1

print('MSE_train is :{}'.format(min(MSE_train24)))
print('MSE_test is :{}'.format(min(MSE_test24)))
print('the best λ for 50(1000-100) set is :{}'.format(MSE_test24.index(min(MSE_test24))))
plt.figure(4)
plt.plot(λ,MSE_train24, color = 'r', label = 'Training')
plt.plot(λ,MSE_test24, color = 'b', label = 'Testing')
plt.title('CV 1000-100 test set')
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.legend()
plt.show()


'''CV 100-100 test set'''

kf = KFold(n_splits = 10)
kf.get_n_splits(train1)
train_x5 = []
test_x5 = []
train_y5 = []
test_y5 = []
for train_index, test_index in kf.split(df2):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_2train5 = df2.drop('y', axis = 1).values
    y_2train5 = df2['y'].values.reshape(-1,1)
    X_train, X_test = X_2train5[train_index], X_2train5[test_index]
    y_train, y_test = y_2train5[train_index], y_2train5[test_index]
    train_x5.append(X_train)
    test_x5.append(X_test)
    train_y5.append(y_train)
    test_y5.append(y_test)

λ = np.array(range(0,151))
R= np.array(range(0,10))
w25 = []
MSE_train25 = []
MSE_test25 = []
n = 0
for u in λ:
    if n < 151:
        m = 0
        u = λ[n]
        w = []
        Mr = []
        Me = []
        for m in R:
            if m < 10 :
                wm = np.dot(np.linalg.inv(train_x5[m].T.dot(train_x5[m]) + u*np.eye(101)), train_x5[m].T).dot(train_y5[m])
                w.append(wm)
                Mrm = ((train_x5[m].dot(wm) - train_y5[m])**2).mean()
                Mr.append(Mrm)
                Mem = ((test_x5[m].dot(wm) - test_y5[m])**2).mean()
                Me.append(Mem)
                m +=1
                
        MSE_train25.append(np.average(Mr))
        MSE_test25.append(np.average(Me))
        w25.append(np.average(w))
        n += 1

print('MSE_train is :{}'.format(min(MSE_train25)))
print('MSE_test is :{}'.format(min(MSE_test25)))
print('the best λ for 100(1000-100) set is :{}'.format(MSE_test25.index(min(MSE_test25))))
plt.figure(5)
plt.plot(λ,MSE_train25, color = 'r', label = 'Training')
plt.plot(λ,MSE_test25, color = 'b', label = 'Testing')
plt.title('CV 100-100 test set')
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.legend()
plt.show()


'''CV 100-10 test set'''

kf = KFold(n_splits = 10)
kf.get_n_splits(train1)
train_x6 = []
test_x6 = []
train_y6 = []
test_y6 = []
for train_index, test_index in kf.split(df3):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_2train6 = df3.drop('y', axis = 1).values
    y_2train6 = df3['y'].values.reshape(-1,1)
    X_train, X_test = X_2train6[train_index], X_2train6[test_index]
    y_train, y_test = y_2train6[train_index], y_2train6[test_index]
    train_x6.append(X_train)
    test_x6.append(X_test)
    train_y6.append(y_train)
    test_y6.append(y_test)

λ = np.array(range(0,151))
R= np.array(range(0,10))
w26 = []
MSE_train26 = []
MSE_test26 = []
n = 0
for u in λ:
    if n < 151:
        m = 0
        u = λ[n]
        w = []
        Mr = []
        Me = []
        for m in R:
            if m < 10 :
                wm = np.dot(np.linalg.inv(train_x6[m].T.dot(train_x6[m]) + u*np.eye(101)), train_x6[m].T).dot(train_y6[m])
                w.append(wm)
                Mrm = ((train_x6[m].dot(wm) - train_y6[m])**2).mean()
                Mr.append(Mrm)
                Mem = ((test_x6[m].dot(wm) - test_y6[m])**2).mean()
                Me.append(Mem)
                m +=1
                
        MSE_train26.append(np.average(Mr))
        MSE_test26.append(np.average(Me))
        w26.append(np.average(w))
        n += 1

print('MSE_train is :{}'.format(min(MSE_train26)))
print('MSE_test is :{}'.format(min(MSE_test26)))
print('the best λ for 150(1000-100) test set is :{}'.format(MSE_test26.index(min(MSE_test26))))
plt.figure(6)
plt.plot(λ,MSE_train26, color = 'r', label = 'Training')
plt.plot(λ,MSE_test26, color = 'b', label = 'Testing')
plt.title('CV 100-10 test set')
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.legend()
plt.show()


'''2-b'''
print('--------------------------------------------------------------------------------')

print('Non-CV the best λ for testing 1000-100 is :{}'.format(MSE_test1.index(min(MSE_test1))))
print('Non-CV the best MSE for testing 1000-100 is :{}'.format(min(MSE_test1)))

print('CV the best λ for 1000-100 train set is :{}'.format(MSE_test21.index(min(MSE_test21))))
print('CV the best MSE for train 1000-100 is :{}'.format(min(MSE_train21)))
print('CV the best MSE for test 1000-100 is :{}'.format(min(MSE_test21)))

print('CV the best λ for 100-100 set is :{}'.format(MSE_test22.index(min(MSE_test22))))
print('CV the best MSE for train 100-100 is :{}'.format(min(MSE_train22)))
print('CV the best MSE for test 100-100 is :{}'.format(min(MSE_test22)))

print('CV the best λ for 100-10 set is :{}'.format(MSE_test23.index(min(MSE_test23))))
print('CV the best MSE for train 100-10 is :{}'.format(min(MSE_train23)))
print('CV the best MSE for test 100-10 is :{}'.format(min(MSE_test23)))

print('CV the best λ for 50(1000-100) is :{}'.format(MSE_test24.index(min(MSE_test24))))
print('CV the best MSE for train 50(1000-100) is :{}'.format(min(MSE_train24)))
print('CV the best MSE for test 50(1000-100) is :{}'.format(min(MSE_test24)))

print('CV the best λ for 100(1000-100) is :{}'.format(MSE_test25.index(min(MSE_test25))))
print('CV the best MSE for train 100(1000-100) is :{}'.format(min(MSE_train25)))
print('CV the best MSE for test 100(1000-100) is :{}'.format(min(MSE_test25)))

print('CV the best λ for 150(1000-100) is :{}'.format(MSE_test26.index(min(MSE_test26))))
print('CV the best MSE for train 150(1000-100) is :{}'.format(min(MSE_train26)))
print('CV the best MSE for test 100(1000-100)is :{}'.format(min(MSE_test26)))

'''2-c'''
print('--------------------------------------------------------------------------------')
'''Time consuming but without considerable improment of MSE!!'''

'''3-a'''
#when λ == 1 
S = np.array(range(1,1001))
R= np.array(range(0,10))
MSE_t1 = []
MSE_te1 = []
wl1 = []
s = 1
X_test1000 = test1.drop('y', axis = 1).values
y_test1000 = test1['y'].values.reshape(-1,1)
for s in S:
    if s < 1001:
        m = 0
        w = []
        Mr = []
        Me = []
        for m in R:
            if m < 10 :
                df = train1.sample(s)
                X_train = df.drop('y', axis = 1).values
                y_train = df['y'].values.reshape(-1,1)
                wm = np.dot(np.linalg.inv(X_train.T.dot(X_train) + np.eye(101)), X_train.T).dot(y_train)
                w.append(wm)
                Mrm = ((X_train.dot(wm) - y_train)**2).mean()
                Mr.append(Mrm)
                Mem = ((X_test1000.dot(wm) - y_test1000)**2).mean()
                Me.append(Mem)
                m +=1 
                   
        MSE_t1.append(np.average(Mr))
        MSE_te1.append(np.average(Me))
        wl1.append(np.average(w))
        s += 1 

#print('MSE_t1 is :{}'.format(min(MSE_t1)))
#print('MSE_e1 is :{}'.format(min(MSE_te1)))
#print('the best λ for training in 100-10 test set is :{}'.format(MSE_t1.index(min(MSE_t1))))
#print('the best λ for testing in 100-10 test set is :{}'.format(MSE_te1.index(min(MSE_te1))))
plt.figure(7,figsize = (10,6))
plt.plot(S,MSE_t1, color = 'r', label = 'Training')
plt.plot(S,MSE_te1, color = 'b', label = 'Testing')
plt.title('Learning Curve λ = 1 ')
plt.xlabel('Number of data')
plt.ylabel('MSE')
plt.legend()
plt.show()


#when λ == 25 
S = np.array(range(1,1001))
R= np.array(range(0,10))
MSE_t2 = []
MSE_te2 = []
wl2 = []
s = 1
X_test1000 = test1.drop('y', axis = 1).values
y_test1000 = test1['y'].values.reshape(-1,1)
for s in S:
    if s < 1001:
        m = 0
        w = []
        Mr = []
        Me = []
        for m in R:
            if m < 10 :
                df = train1.sample(s)
                X_train = df.drop('y', axis = 1).values
                y_train = df['y'].values.reshape(-1,1)
                wm = np.dot(np.linalg.inv(X_train.T.dot(X_train) + 25*np.eye(101)), X_train.T).dot(y_train)
                w.append(wm)
                Mrm = ((X_train.dot(wm) - y_train)**2).mean()
                Mr.append(Mrm)
                Mem = ((X_test1000.dot(wm) - y_test1000)**2).mean()
                Me.append(Mem)
                m += 1 
                   
        MSE_t2.append(np.average(Mr))
        MSE_te2.append(np.average(Me))
        wl2.append(np.average(w))
        s += 1 

#print('MSE_t1 is :{}'.format(min(MSE_t1)))
#print('MSE_e1 is :{}'.format(min(MSE_te1)))
#print('the best λ for training in 100-10 test set is :{}'.format(MSE_t1.index(min(MSE_t1))))
#print('the best λ for testing in 100-10 test set is :{}'.format(MSE_te1.index(min(MSE_te1))))
plt.figure(8,figsize = (10,6))
plt.plot(S,MSE_t2, color = 'r', label = 'Training')
plt.plot(S,MSE_te2, color = 'b', label = 'Testing')
plt.title('Learning Curve λ = 25 ')
plt.xlabel('Number of data')
plt.ylabel('MSE')
plt.legend()
plt.show()


#when λ == 150 
S = np.array(range(1,1001))
R= np.array(range(0,10))
MSE_t3 = []
MSE_te3 = []
wl3 = []
s = 1
X_test1000 = test1.drop('y', axis = 1).values
y_test1000 = test1['y'].values.reshape(-1,1)
for s in S:
    if s < 1001:
        m = 0
        w = []
        Mr = []
        Me = []
        for m in R:
            if m < 10 :
                df = train1.sample(s)
                X_train = df.drop('y', axis = 1).values
                y_train = df['y'].values.reshape(-1,1)
                wm = np.dot(np.linalg.inv(X_train.T.dot(X_train) + 150*np.eye(101)), X_train.T).dot(y_train)
                w.append(wm)
                Mrm = ((X_train.dot(wm) - y_train)**2).mean()
                Mr.append(Mrm)
                Mem = ((X_test1000.dot(wm) - y_test1000)**2).mean()
                Me.append(Mem)
                m += 1 
                   
        MSE_t3.append(np.average(Mr))
        MSE_te3.append(np.average(Me))
        wl3.append(np.average(w))
        s += 1 

#print('MSE_t3 is :{}'.format(min(MSE_t3)))
#print('MSE_e3 is :{}'.format(min(MSE_te3)))
#print('the best λ for training in 100-10 test set is :{}'.format(MSE_t3.index(min(MSE_t1))))
#print('the best λ for testing in 100-10 test set is :{}'.format(MSE_te3.index(min(MSE_te1))))
plt.figure(7,figsize = (10,6))
plt.plot(S,MSE_t3, color = 'r', label = 'Training')
plt.plot(S,MSE_te3, color = 'b', label = 'Testing')
plt.title('Learning Curve λ = 150 ')
plt.xlabel('Number of data')
plt.ylabel('MSE')
plt.legend()
plt.show()


λ4 = [1,25,150]
datapoint = np.arange(20,1001)

λMSE4 =[]
λtest_MSE4 = []
n = 11

for u in λ4:
    iMSE4 = []
    itest_MSE4 = []
    for i in datapoint:
        MSE4 = []
        W4 = []
        test_MSE4 = []
        for j in range(n):
            train4 = train1.sample(i)
            train4_DFX = train4.iloc[:,0:101]
            train4_DFY = train4.iloc[:,101]
            train4_X = train4_DFX.values
            train4_Y = train4_DFY.values
            train4_XT = np.transpose(train4_X)
        
            S4 = np.dot(train4_XT,train4_X)+u*np.eye(101)
            w4 = np.dot(np.dot((np.linalg.inv(S4)),train4_XT),train4_Y)
            mse4 = (np.dot(train4_X,w4)-train4_Y)**2
            test_mse4 = (np.dot(X_test1000,w4)-y_test1000)**2
            W4.append(w4)
            MSE4.append(np.average(mse4))
            test_MSE4.append(np.average(test_mse4))
        iMSE4.append(np.average(MSE4))
        itest_MSE4.append(np.average(test_MSE4))
    λMSE4.append(iMSE4)
    λtest_MSE4.append(itest_MSE4)

plt.figure(8,figsize = (10,6))
plt.plot(datapoint,λMSE4[0], color = 'r', label = 'Training')
plt.plot(datapoint,λtest_MSE4[0], color = 'b', label = 'Testing')
plt.title('Learning Curve λ = 1 ')
plt.xlabel('Number of data')
plt.ylabel('MSE')
plt.legend()
plt.show()

plt.figure(9,figsize = (10,6))
plt.plot(datapoint,λMSE4[1], color = 'r', label = 'Training')
plt.plot(datapoint,λtest_MSE4[1], color = 'b', label = 'Testing')
plt.title('Learning Curve λ = 25 ')
plt.xlabel('Number of data')
plt.ylabel('MSE')
plt.legend()
plt.show()

plt.figure(10,figsize = (10,6))
plt.plot(datapoint,λMSE4[2], color = 'r', label = 'Training')
plt.plot(datapoint,λtest_MSE4[2], color = 'b', label = 'Testing')
plt.title('Learning Curve λ = 150 ')
plt.xlabel('Number of data')
plt.ylabel('MSE')
plt.legend()
plt.show()

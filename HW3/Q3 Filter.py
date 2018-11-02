
# coding: utf-8

# In[304]:


import pandas as pd
import math
import numpy as np
from scipy.io import arff
import heapq
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


# In[482]:


data = arff.loadarff('veh-prime.arff')
df = pd.DataFrame(data[0])


# In[483]:


# getting the dataframe from weka!!
df.head()


# In[484]:


#getting the features from the dataframe
features = df.columns.drop(df.columns[-1])
#getting the target
train_targets = df[['CLASS']]
old_labels = [b'noncar',b'car']
new_labels = [0,1]
#transforming the target to 0-1 and its type is data frame.
train_n_targets = train_targets.replace(old_labels,new_labels)
train_n_targets.head()


# In[391]:


#getting the distance matrix for every test data
def get_distance(X_train,X_test):
    matrix_distance = []
    for i in range(X_test.shape[0]):
        list_distance = []
        for j in range(X_train.shape[0]):
            d = (np.sum((X_test[i] - X_train[j])**2))**0.5
            list_distance.append(d)
        matrix_distance.append(list_distance)
    matrix_distance = pd.DataFrame(matrix_distance)
    return matrix_distance

#Comparing the accuracy
def score(ypred,testing):
    return (list(np.array(ypred) - np.array(testing)).count(0))/len(ypred)  


# In[401]:


# 1 the matrix,2 k ,3 , train_targets
def get_k_outcome(matrix_distance,k,train_targets,test_traget):
    test_outcome = []
    x, y = matrix_distance.shape
    for i in k:
        list_test_target = []
        Y = []
        for j in range(x) :
            instances = matrix_distance.iloc[j]
            test_target = heapq.nsmallest(i,instances)
            min_num_index_list = list(map(list(instances).index, test_target))
            list_test_target.append(min_num_index_list)
            
        for n in list_test_target :
            t = train_targets[n]
            t_array = np.array(t)
            if t_array.sum()*2 < i :
                y = int(0)
            else :
                y = int(1)
            Y.append(y)
        test_outcome = score(Y,test_traget)
    return test_outcome    # its knn's prediction accuracy    


# In[351]:


#Pearson product-moment correlation coefficient
def get_ppc(feature, target_tocheck):
    sum_f = 0
    sum_t = 0
    product = 0
    mean_f = 0
    mean_t = 0
    target_tocheck = list(target_tocheck)
    for i in range(len(feature)):
        sum_f += float(feature[i])**2
        sum_t += float(target_tocheck[i])**2
        product += float(feature[i])* float(target_tocheck[i])
        mean_f += float(feature[i])
        mean_t += float(target_tocheck[i])
        
    mean_f = mean_f/len(feature)
    mean_t = mean_t/len(feature)
    pop_sd_f = ((sum_f/len(feature)) - (mean_f **2))**0.5
    pop_sd_t = ((sum_t/len(feature)) - (mean_t **2))**0.5  
    cov = (product / len(feature)) - (mean_f * mean_t)
    correlation = cov / (pop_sd_t * pop_sd_f)
    return correlation


# In[339]:


f1_y = []
for j in features:
    y = abs(get_ppc(df[j], target_tocheck = train_n_targets['CLASS']))
    f1_y.append(y)
index = list(range(36))

c={"Feature" : features,
   "Pcc" : f1_y}
#getting the features' PCC
level_1 = pd.DataFrame(c)
print(level_1)


# In[340]:


#Sorting
ranking = level_1.sort_values(by='Pcc',ascending=False,)
print(ranking)


# In[341]:


#getting the list of oerdered feature
filter_feature = ranking['Feature']
f_feature = []
for i in filter_feature:
    f_feature.append(i)
print(f_feature)


# In[342]:


#Normalization the data
df = df[features]
zdf = (df - df.mean())/df.std()
zdf.head()


# In[344]:


# normalizing get the same result
fz_y = []
for j in features:
    y = abs(get_ppc(zdf[j], target_tocheck = train_n_targets['CLASS']))
    fz_y.append(y)
z={"Feature" : features,
   "Pcc" : fz_y}
level_z = pd.DataFrame(z)
print(level_z)


# In[404]:


list_ACC = []
for j in list(range(1,37,1)):
    loo = LeaveOneOut()
    X = zdf[f_feature[0:j]]
    train_x = []
    test_x = []
    train_y = []
    test_y = []
    loo.get_n_splits(X)
    for train_index, test_index in loo.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        y = np.array(train_n_targets).reshape(-1,1)
        X_train, X_test = X.iloc[train_index].values, X.iloc[test_index].values
        y_train, y_test = y[train_index], y[test_index]
        #print(X_train, X_test, y_train, y_test)
        train_x.append(X_train)
        test_x.append(X_test)
        train_y.append(y_train)
        test_y.append(y_test)
    S = []
    k = [7]
    for i in range(len(train_x)):
        _ = get_distance(train_x[i],test_x[i])
        #print(_)
        Y = get_k_outcome(_,k,train_y[i],test_y[i])
        #print(Y)
        S.append(Y)
        #print(S)
    Ave_acc = np.mean(S)
    print(Ave_acc)
    list_ACC.append(Ave_acc)
print(list_ACC)


# In[521]:


print(max(list_ACC))
Final = {"Feature" : f_feature,
   "Acc" : list_ACC}
final = pd.DataFrame(Final)
print(final)


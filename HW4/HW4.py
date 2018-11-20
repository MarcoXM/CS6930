
# coding: utf-8

# In[458]:


import pandas as pd
import math
import numpy as np
from scipy.io import arff
import heapq
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from collections import defaultdict


# In[459]:


data = arff.loadarff('segment.arff')
df = pd.DataFrame(data[0])
# getting the dataframe from weka!!
df.shape


# In[460]:


#getting the features from the dataframe
features = df.columns.drop(df.columns[-1])
#Normalization the X
df1 = df[features]
zdf = (df1 - df1.mean())/df1.std()
zdf.head()
zdf = zdf.fillna(0)
a = [775, 1020, 200, 127, 329]
b = [1626, 1515, 651, 658, 328]
c = [775, 1020, 200, 127, 329, 1626, 1515, 651, 658, 328, 1160, 108, 422, 88, 105, 261, 212,
1941, 1724, 704, 1469, 635, 867, 1187, 445, 222, 1283, 1288, 1766, 1168, 566, 1812, 214,
53, 423, 50, 705, 1284, 1356, 996, 1084, 1956, 254, 711, 1997, 1378, 827, 1875, 424,
1790, 633, 208, 1670, 1517, 1902, 1476, 1716, 1709, 264, 1, 371, 758, 332, 542, 672, 483,
65, 92, 400, 1079, 1281, 145, 1410, 664, 155, 166, 1900, 1134, 1462, 954, 1818, 1679,
832, 1627, 1760, 1330, 913, 234, 1635, 1078, 640, 833, 392, 1425, 610, 1353, 1772, 908,
1964, 1260, 784, 520, 1363, 544, 426, 1146, 987, 612, 1685, 1121, 1740, 287, 1383, 1923,
1665, 19, 1239, 251, 309, 245, 384, 1306, 786, 1814, 7, 1203, 1068, 1493, 859, 233, 1846,
1119, 469, 1869, 609, 385, 1182, 1949, 1622, 719, 643, 1692, 1389, 120, 1034, 805, 266,
339, 826, 530, 1173, 802, 1495, 504, 1241, 427, 1555, 1597, 692, 178, 774, 1623, 1641,
661, 1242, 1757, 553, 1377, 1419, 306, 1838, 211, 356, 541, 1455, 741, 583, 1464, 209,
1615, 475, 1903, 555, 1046, 379, 1938, 417, 1747, 342, 1148, 1697, 1785, 298, 1485,
945, 1097, 207, 857, 1758, 1390, 172, 587, 455, 1690, 1277, 345, 1166, 1367, 1858, 1427,
1434, 953, 1992, 1140, 137, 64, 1448, 991, 1312, 1628, 167, 1042, 1887, 1825, 249, 240,
524, 1098, 311, 337, 220, 1913, 727, 1659, 1321, 130, 1904, 561, 1270, 1250, 613, 152,
1440, 473, 1834, 1387, 1656, 1028, 1106, 829, 1591, 1699, 1674, 947, 77, 468, 997, 611,
1776, 123, 979, 1471, 1300, 1007, 1443, 164, 1881, 1935, 280, 442, 1588, 1033, 79, 1686,
854, 257, 1460, 1380, 495, 1701, 1611, 804, 1609, 975, 1181, 582, 816, 1770, 663, 737,
1810, 523, 1243, 944, 1959, 78, 675, 135, 1381, 1472]
idice = a+b+c
X = zdf.values


# In[461]:


zdf.count()


##getting the distance matrix for every test data
def get_distance(X_train,random_points):
    matrix_distance = []
    for i in range(X_train.shape[0]):
        list_distance = []
        for j in random_points:
            d = (np.sum((X_train[i] - j)**2))**0.5
            list_distance.append(d)
        matrix_distance.append(list_distance)
    return matrix_distance

# 1 the matrix,2 k ,3 , train_targets
def get_group(matrix_distance):
    outcome = []
    for j in range(len(matrix_distance)) :
        index_dice = matrix_distance[j].index(min(matrix_distance[j]))
        outcome.append(index_dice)
    return outcome

def get_set(tags):
    d = defaultdict(list)
    n = 0
    for i in tags:
        d[i].append(n)
        n = n + 1
    return d

#np.linalg.norm([all_features[j] - final_centroids[i]], axis = 1)
def get_SSE(group_point):
    s = 0
    for i in group_point:
        a=np.sum((X[group_point[i]]-initial[i])**2)
        s+= a
    return s

def get_newcentroids(X_train_dataframe,group_point):
    da = []
    for l in sorted(group_point.keys()):
        a = X_train_dataframe.iloc[group_point[l],:].mean()
        a = np.array(a)
        da.append(a)
    return np.array(da)

k_list = list(range(1,13)) #setting the k from 1 to 12
every_k_mean = []
every_k_std = []
for k in k_list:    
    idice_list = idice.copy()
    idice_list = idice_list[:(25*k)] # getting the start points only could run 25 times 
    #print(len(idice_list))
    best_cluster = [] #for each condition, we get the final/lowest SSE
    while len(idice_list) != 0: # picking the points to iterate
        a = [] 
        for z in range(k):
            v = idice_list.pop(0)
            a.append(v)
        #getting the initial seed for the iteration
        #print(len(idice_list))
        #print(a)
        initial = [] # 
        for initial_centroid in a:
            initial.append(X[initial_centroid])#
        initial = np.array(initial)
        #print(initial)
        #print('initial completion !!!!')

        outcome_group = []
        outcome_score = []
        for round_try in range(50): #interation is a 50 instance
            seed = initial.copy()
            #print(seed)
            dis = get_distance(X,seed)
            run = get_group(dis)
            #print(run)
            group_points = get_set(run)
            
            #print(group_points)
            score = get_SSE(group_points)
            #print(score)
            next_seed = get_newcentroids(zdf,group_points)
            #print(seed)
            #print(next_seed)
            if np.array_equal(seed,next_seed)== False:
                initial = next_seed.copy()
                outcome_group.append(group_points)
                outcome_score.append(score)
            else:
                outcome_group.append(group_points)
                outcome_score.append(score)
                break
        print(outcome_score)    
        best = min(outcome_score)# get the lowest SSE, with The final centroid for specific start points.
        best_cluster.append(best)# getting different answer which starts in others points
    mean_SSE = np.mean(best_cluster) #getting the mean
    std_SSE = np.std(best_cluster)#and std between
    print('when k is ',k,'the mean SSE is',mean_SSE)
    every_k_mean.append(mean_SSE)
    print('when k is ',k,'the std SSE is',std_SSE)
    every_k_std.append(std_SSE)
print('finish')


column_names = ['k', 'µk', 'µk − 2σk','µk + 2σk']
low = np.array(every_k_mean) - np.array(every_k_std)*2
low = list(low)
high = np.array(every_k_mean) + np.array(every_k_std)*2
high = list(high)
answer = {'k':k_list,
          'µk':every_k_mean,
          'µk − 2σk':low,
          'µk + 2σk':high}
answer_df = pd.DataFrame(answer)
answer_df

#Ploting Graphes
sns.set_style('darkgrid')
fig, ax = plt.subplots()
plt.plot(answer_df['k'],answer_df['µk'])
ax.set(title = 'K and its SSE')
plt.xlabel = 'K'
plt.ylabel = 'SSE'
plt.ylim = (-1,50000)
plt.plot([1,1],[41562.000000,41562.000000],linewidth = 5)
plt.plot([2,2],[25065.114324,33344.915966],linewidth = 5)
plt.plot([3,3],[22822.613700,25765.466540],linewidth = 5)
plt.plot([4,4],[18074.079511,23806.877159],linewidth = 5)
plt.plot([5,5],[14973.819738,21738.137267],linewidth = 5)
plt.plot([6,6],[12957.018588,18633.290350],linewidth = 5)
plt.plot([7,7],[11406.142545,17214.849648],linewidth = 5)
plt.plot([8,8],[11009.332892,15547.067105],linewidth = 5)
plt.plot([9,9],[10792.188185,13845.362009],linewidth = 5)
plt.plot([10,10],[10266.127553,12824.898736],linewidth = 5)
plt.plot([11,11],[9802.077880,12050.751093],linewidth = 5)
plt.plot([12,12],[9387.465628,11753.850751],linewidth = 5)
plt.show()





# In[ ]:





# In[ ]:





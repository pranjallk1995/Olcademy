# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 08:17:00 2019

@author: Pranjall

Assumpitons: 1.) A retail store wishes to identify types of potential customers in its vicinity and stock its inventory according to their needs to reduce unnecessary spending.
                 eg: if there are very few south asian origin people in the region, stocking on fairness products is unnecessary unless they have high income.
             
             2.) For this the retail store can accquire data from other companies in the region to get insights of the demography of the region. 
                
Aim: To group customers into two classes:
                                        1.) Require door to door promotion. (eg. High income and elderly)
                                        2.) Require discounts and offers.   (eg. Low income and young)
"""

#importing libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#importing data.
dataset = pd.read_csv("core_dataset.csv")

#dropping unnecessary attributes.
dataset = dataset[['State', 'Age', 'Sex', 'MaritalDesc', 'Hispanic/Latino', 'RaceDesc', 'Pay Rate']]

#counting frequency of attributes.
Sex = dataset.groupby('Sex').count()
State = dataset.groupby('State').count()
Marital = dataset.groupby('MaritalDesc').count()
Hispanic = dataset.groupby('Hispanic/Latino').count()
Race = dataset.groupby('RaceDesc').count()

#converting into numpy arrays.
Age = dataset.iloc[:, 1]
Age = np.array(Age)
l = len(Age)
Age = Age.reshape(l, 1)

Income = dataset.iloc[:, 6]
Income = np.array(Income)
l = len(Income)
Income = Income.reshape(l, 1)

#plotting histograms and bar-graphs
fig, ax1 = plt.subplots(2, 2)

ax1[0, 0].bar(Sex.index, Sex.iloc[:, 0])
ax1[0, 0].set_title("Sex")
ax1[0, 0].grid()

ax1[1, 0].bar(Marital.index, Marital.iloc[:, 0])
ax1[1, 0].set_title("Marital Status")
ax1[1, 0].grid()

ax1[0, 1].bar(Hispanic.index, Hispanic.iloc[:, 0])
ax1[0, 1].set_title("Hispanic/Latino")
ax1[0, 1].grid()

ax1[1, 1].hist(Age, bins = 50)
ax1[1, 1].set_title("Age")
ax1[1, 1].grid()

fig, ax2 = plt.subplots(2, 1)

ax2[0].bar(State.index, State.iloc[:, 0])
ax2[0].set_title("Original state")
ax2[0].grid()

ax2[1].bar(Race.index, Race.iloc[:, 0])
ax2[1].set_title("Ethnicity")
ax2[1].grid()

#plotting scatter-plots and box-plots.
fig, ax = plt.subplots()
ax.scatter(dataset['Age'], dataset['Pay Rate'])
ax.set_title('Age wise distibution of income')
ax.set_xlabel('Age')
ax.set_ylabel('Income')
ax.grid()

dataset.boxplot(column = 'Pay Rate', by = 'Sex')
dataset.boxplot(column = 'Pay Rate', by = 'RaceDesc')
dataset.boxplot(column = 'Pay Rate', by = 'MaritalDesc')
dataset.boxplot(column = 'Pay Rate', by = "Hispanic/Latino")


#modifying dataset to implement grouping of customers based on age and income.
dataset = dataset[['Age', 'Pay Rate']]

#normalizing data.
from sklearn.preprocessing import StandardScaler
dataset_sc = StandardScaler()
dataset = dataset.astype('float')
dataset = pd.DataFrame(dataset_sc.fit_transform(dataset))

#converting to numpy array.
data = np.array(dataset)

#monitoring execution time.
start_time = time.time()

#applying K-means.
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
pred = kmeans.fit_predict(data)

#storing execution time.
execution_time = time.time() - start_time

#Visualizing K-means.
fig, ax3 = plt.subplots()
ax3.scatter(data[pred == 0, 0], data[pred == 0, 1], c = 'red', label = 'Cluster 1')
ax3.scatter(data[pred == 1, 0], data[pred == 1, 1], c = 'blue', label = 'Cluster 2')
ax3.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 30, c = 'black', label = 'Centroid')
ax3.set_title("Customer Groupings")
ax3.set_xlabel("Age")
ax3.set_ylabel("Income")
ax3.legend()
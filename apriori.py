#!/usr/bin/env python
# coding: utf-8

# # Apriori
# 
# #### Author: Chaitanya Laxman

# ## Importing the libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Data Preprocessing

# In[4]:


dataset = pd.read_csv('data.csv', header = None)
dataset.shape


# In[5]:


dataset.head()


# In[6]:


transactions = []
for i in range(7501):
    temp = []
    for j in range(20):
        temp.append(str(dataset.values[i,j]))
    transactions.append(temp)


# In[7]:


len(transactions)


# ## Training the Apriori model on the dataset

# In[8]:


from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)


# ## Visualising the results

# In[9]:


results = list(rules)


# In[10]:


results


# In[11]:


results[0]


# In[12]:


def extract(results):
    
    item1 = []
    for i in results:
        item1.append(tuple(i[2][0][0])[0])
    
    item2 = []
    for i in results:
        item2.append(tuple(i[2][0][1])[0])
    
    supports = []
    for i in results:
        supports.append(i[1])
    
    confidences = []
    for i in results:
        confidences.append(i[2][0][2])
    
    lifts = []
    for i in results:
        lifts.append(i[2][0][2])
    
    return list(zip(item1, item2, supports, confidences, lifts))


# In[13]:


dataframe = pd.DataFrame(extract(results), columns = ["Item1", "Item2", "Supports", "Confidences", "Lifts"])
dataframe.head()


# In[14]:


dataframe.nlargest(n = 15, columns = "Lifts")


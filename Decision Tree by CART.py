#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv("Social_Network_Ads (1).csv")


# In[4]:


df


# In[5]:


x = df.iloc[:,[2,3]].values


# In[6]:


y = df.iloc[:,4].values


# In[7]:


x


# In[8]:


y


# ## Train Test Split

# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 0)


# In[11]:


x_train


# In[12]:


y_train


# In[13]:


x_test


# In[14]:


y_test


# ## Apply Feature Scalling

# In[15]:


from sklearn.preprocessing import StandardScaler


# In[16]:


sc = StandardScaler()


# In[17]:


x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# In[18]:


x_train


# In[19]:


x_test


# ## making Decision Tree model with the help of Cart

# In[20]:


from sklearn.tree import DecisionTreeClassifier


# In[21]:


dtclr = DecisionTreeClassifier(criterion = "gini", splitter = "random", random_state = 0)


# In[22]:


dtclr.fit(x_train, y_train)


# In[23]:


y_pred = dtclr.predict(x_test)


# In[24]:


y_pred


# In[25]:


y_test


# ## Visualize the tested and predicted data

# In[26]:


x_test[:,0]


# In[27]:


plt.scatter(x_test[:,0],y_test, c = y_pred)


# ## calculating accuracy and confusion matrix

# In[28]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[29]:


"accuracy_score : ", accuracy_score(y_test, y_pred)


# In[30]:


cf = confusion_matrix(y_test, y_pred)


# In[31]:


cf


# In[ ]:





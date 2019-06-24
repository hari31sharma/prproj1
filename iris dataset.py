#!/usr/bin/env python
# coding: utf-8

# In[5]:


from sklearn import datasets
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np


# In[6]:


iris= datasets.load_iris()


# In[7]:


#print(iris.DESCR)


# In[8]:


import pandas as pd


# In[9]:


df= pd.DataFrame(iris['data'],columns=iris['feature_names'])


# In[10]:


df['species']=iris.target


# In[11]:


df.head()


# In[12]:


df.plot()


# In[13]:


sns.heatmap(data=iris['data'])


# In[14]:


sns.heatmap(df.corr(),annot=True)


# In[15]:


sns.scatterplot(data=df.drop('species',axis=1),hue='species')


# In[16]:


iris.target_names


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


real_x=df.iloc[:,:3].values
real_y=df.iloc[:,4].values


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(real_x, real_y, test_size=0.55, random_state=42)


# In[20]:


from sklearn.svm import SVC


# In[29]:


mod=SVC(kernel='linear',C=100,gamma=0.001)


# In[30]:


mod.fit(X_train,y_train)


# In[31]:


pred=mod.predict(X_test)


# In[32]:


from sklearn.metrics import confusion_matrix,classification_report


# In[33]:


print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[40]:


from sklearn import metrics


# In[46]:


fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=1)


# In[47]:


metrics.auc(fpr, tpr)


# In[41]:


metrics.auc(y_test,pred)


# In[ ]:





# In[ ]:





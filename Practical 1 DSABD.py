#!/usr/bin/env python
# coding: utf-8

# In[62]:


import numpy as np 
import pandas as pd 
from sklearn import preprocessing


# In[63]:


#Importing the file 
df=pd.read_csv("D:\Lalit Pawar Practicals\spambase_csv.csv")


# In[64]:


df


# In[65]:


#Indexing 
df.index


# In[66]:


#data visualization
df.head(n=10)


# In[67]:


# SHAPE TO GET no of rows and columns
df.shape 


# In[68]:


#To check whether the target attribute is binary or not 
np.unique(df['class'])


# In[69]:


df.columns.values


# In[70]:


df['word_freq_address']


# In[71]:


df.isnull()


# In[72]:


df.iloc


# In[73]:


df.isnull().ne


# In[74]:


df.isnull().sum().sum()


# In[75]:


df.isnull().sum(axis=1)


# In[76]:


df.word_freq_remove.isnull().sum()


# In[77]:


df.groupby(['word_freq_address'])['word_freq_make'].apply(lambda x:x.isnull().sum())


# In[78]:


df.describe(include='all')


# In[79]:


df.sort_index(axis=1,ascending=False)

df.sort_values(by="word_freq_address")
# In[80]:


df.iloc[5]


# In[81]:


df[0:3]


# In[82]:


df.loc[:,["word_freq_make","word_freq_address"]]


# In[83]:


df.iloc[:4,:]


# In[84]:


df.iloc[:, :4]


# In[85]:


df.iloc[:4, :6]


# In[86]:


df.iloc[3:5, 0:2]


# In[87]:


df.iloc[[1,2,4],[0,2]]


# In[88]:


df.iloc[3:5, :]


# In[89]:


df.iloc[:, 1:3]


# In[90]:


df.iloc[1:1]


# In[91]:


col_1_4=df.columns[1:4]


# In[92]:


df[df.columns[2:4]].iloc[5:10]


# In[93]:


min_max_scaler=preprocessing.MinMaxScaler()


# In[94]:


x=df.iloc[:,:4]


# In[95]:


x_scaled=min_max_scaler.fit_transform(x)


# In[96]:


df_normalized= pd.DataFrame(x_scaled)


# In[97]:


df_normalized


# In[98]:


df.head


# In[ ]:





# In[ ]:





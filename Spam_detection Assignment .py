#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer



# In[16]:


#Importing the file 
df=pd.read_csv("D:\Machine Learning project\spam.csv",encoding="latin-1")


# In[18]:


#data visualization
df.head(n=10)


# In[19]:


df.shape 


# In[20]:


#To check whether the target attribute is binary or not 
np.unique(df['class'])


# In[21]:


np.unique(df['message'])


# In[22]:


# creating Sparse matrix 
x=df["message"].values
y=df["class"].values

# Create CountVectorizer object .
cv=CountVectorizer()  # count vectorizer is converting text into sparse matrix

x=cv.fit_transform(x)
v=x.toarray()

print(v)


# In[23]:


first_col=df.pop('message')
df.insert(0,'message',first_col)  #shifting mesg column to first column
df


# In[32]:


# Spliting 3:1
# Training and Testing

train_x= x[:4180]
train_y= y[:4180]

test_x=x[4180:]
test_y=y[4180:]


# In[35]:


bnb=BernoulliNB(binarize=0.0)
model=bnb.fit(train_x,train_y)

y_pred_train=bnb.predict(train_x)
y_pred_test=bnb.predict(test_x)


# In[36]:


#printing training score 
print(bnb.score(train_x,train_y)*100)

#printing testing score 
print(bnb.score(test_x,test_y)*100)


# In[37]:


from sklearn.metrics import classification_report

print(classification_report(train_y,y_pred_train))


# In[38]:


from sklearn.metrics import classification_report

print(classification_report(test_y,y_pred_test))


# In[ ]:





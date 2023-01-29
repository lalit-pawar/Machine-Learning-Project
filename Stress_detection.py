#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd 


# In[3]:


df=pd.read_csv('D:\Machine Learning project\stress.csv')


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[8]:


df.columns


# In[7]:


import nltk
import re
from nltk. corpus import stopwords
import string
nltk. download( 'stopwords' )
stemmer = nltk. SnowballStemmer("english")
stopword=set (stopwords . words ( 'english' ))

def clean(text):
    text = str(text) . lower()  #returns a string where all characters are lower case. Symbols and Numbers are ignored.
    text = re. sub('\[.*?\]',' ',text)  #substring and returns a string with replaced values.
    text = re. sub('https?://\S+/www\. \S+', ' ', text)#whitespace char with pattern
    text = re. sub('<. *?>+', ' ', text)#special char enclosed in square brackets
    text = re. sub(' [%s]' % re. escape(string. punctuation), ' ', text)#eliminate punctuation from string
    text = re. sub(' \n',' ', text)
    text = re. sub(' \w*\d\w*' ,' ', text)#word character ASCII punctuation
    text = [word for word in text. split(' ') if word not in stopword]  #removing stopwords
    text =" ". join(text)
    text = [stemmer . stem(word) for word in text. split(' ') ]#remove morphological affixes from words
    text = " ". join(text)
    return text
df [ "text"] = df["text"]. apply(clean)


# In[17]:


import matplotlib. pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
text = " ". join(i for i in df. text)
stopwords = set (STOPWORDS)
wordcloud = WordCloud( stopwords=stopwords,background_color="black") . generate(text)
plt. figure(figsize=(10, 15) )
plt. imshow(wordcloud )
plt. axis("off")
plt. show( )


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 

x= np.array (df["text"])
y= np.array (df["label"])

cv= CountVectorizer()
X= cv.fit_transform(x)

print(X)

xtrain, xtest, ytrain, ytest, = train_test_split(X,y,test_size=0.30,random_state=40)


# In[10]:


from sklearn.naive_bayes import BernoulliNB
model=BernoulliNB()
model.fit(xtrain,ytrain)


# In[11]:


user = input(" Enter the text ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)







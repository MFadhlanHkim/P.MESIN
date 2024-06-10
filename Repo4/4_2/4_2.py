#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Author: Lea Kotler
# Date: 6 July 2023

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


df = pd.read_csv("Uncleaned_Cnn.csv")
df1 = pd.read_csv("Uncleaned_cnn1.csv")
df2 = pd.read_csv("Uncleaned_cnn_week1.csv")


# In[4]:


print(df.info())
print(df1.info())
print(df2.info())


# In[5]:


print(df.shape)
print(df1.shape)
print(df2.shape)


# In[6]:


duplicates = [pd.merge(df, df1,on=['title'],how ='inner')['title'].to_list()]
duplicates.append(pd.merge(df, df2,on=['title'],how ='inner')['title'].to_list())
duplicates.append(pd.merge(df1, df2,on=['title'],how ='inner')['title'].to_list())
print(duplicates)


# In[7]:


frames = [df,df1,df2]
df_merge = pd.concat(frames,ignore_index=True)
df_merge


# In[8]:


#Splitting "views" column into columns "number" and "text"
df_merge[['number with prefix','text']]=df_merge['views'].str.split(expand=True)
df_merge['number']=df_merge['number with prefix'].str[:-1]
df_merge['prefix']=df_merge['number with prefix'].str[-1:]
df_merge


# In[9]:


df_merge['prefix'].unique()


# In[10]:


df_merge['K'] = df_merge[df_merge.prefix=='K']['number'].astype(float)*1000
df_merge['M'] = df_merge[df_merge.prefix=='M']['number'].astype(float)*1000000
df_merge['num'] = df_merge[df_merge.prefix=='3']['number'].astype(float)*10 + 3
df_merge['number of views'] = df_merge['K'].combine_first(df_merge['M'].combine_first(df_merge['num']))

df_merge[~(df_merge.prefix=='K')].head(5)


# In[11]:


del df_merge['number with prefix']
del df_merge['text']
del df_merge['number']
del df_merge['prefix']
del df_merge['K']
del df_merge['M']
del df_merge['num']


# In[12]:


df_merge.head()


# In[13]:


#Splitting "views" column into columns "number" and "text"
df_merge[['number','prefix','text']]=df_merge['when'].str.split(expand=True)
df_merge


# In[14]:


df_merge['prefix'].unique()


# In[15]:


df_merge[~(df_merge.prefix=='minutes')]


# In[16]:


df_merge['minutes'] = df_merge[df_merge.prefix=='minutes']['number'].astype(int)
df_merge['hours'] = df_merge[(df_merge.prefix=='hour')|(df_merge.prefix=='hours')]['number'].astype(int)*60
df_merge['days'] = df_merge[(df_merge.prefix=='day')| (df_merge.prefix=='days')]['number'].astype(int)*60*24
df_merge['weeks'] = df_merge[(df_merge.prefix=='week')| (df_merge.prefix=='weeks')]['number'].astype(int)*60*24*7
df_merge['months'] = df_merge[(df_merge.prefix=='month')| (df_merge.prefix=='months')]['number'].astype(int)*60*24*30
df_merge['when viewed in minutes'] = df_merge['minutes'].combine_first(df_merge['hours'].combine_first(df_merge['days'].combine_first(df_merge['weeks'].combine_first(df_merge['months']))))
df_merge


# In[17]:


del df_merge['text']
del df_merge['number']
del df_merge['prefix']
del df_merge['minutes']
del df_merge['hours']
del df_merge['days']
del df_merge['weeks']
del df_merge['months']


# In[18]:


df_merge = df_merge.sort_values('number of views', ascending=False)
df_merge


# In[19]:


df_merge = df_merge.drop_duplicates(subset=['title'])
df_merge


# In[20]:


f, ax = plt.subplots(figsize=(15,7))
plt.scatter(df_merge['number of views'], df_merge['when viewed in minutes'])
ax.set_xlabel('number of views')
ax.set_ylabel('when viewed in minutes')
plt.show()


# In[21]:


df_merge.head(10)


# In[22]:


df_merge[df_merge['when viewed in minutes']<=60*24].head(10)


# In[23]:


stopwords = STOPWORDS
print(stopwords)


# In[24]:


text_data=" ".join(df_merge['title'])
wrdcld=WordCloud(width=800,height=400,background_color='white').generate(text_data)


# In[25]:


plt.figure(figsize=(10, 6))
plt.imshow(wrdcld, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:





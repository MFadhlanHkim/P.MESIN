#!/usr/bin/env python
# coding: utf-8

# In[5]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[6]:


pip install fanalysis


# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


data_cnn = pd.read_csv('Uncleaned_Cnn.csv').drop_duplicates()
data_cnn.head()


# In[13]:


data_cnn.info()


# In[14]:


data_cnn['when'].value_counts()


# In[15]:


def month_grouping(date):
    if date == '4 months ago':
        return '4 months ago'
    elif date == '3 months ago':
        return '3 months ago'
    elif date == '2 months ago':
        return '2 months ago'
    elif date == '1 month ago':
        return '1 months ago'
    else:
        return 'this month'


# In[16]:


data_cnn['month_group'] = data_cnn['when'].apply(lambda x: month_grouping(x))
data_cnn.head()


# In[17]:


data_cnn['month_group'] = pd.Categorical(data_cnn['month_group'], categories=['4 months ago', '3 months ago', '2 months ago', '1 months ago', 'this month'], ordered=True)
data_cnn.info()


# In[18]:


def extract_view(view):
    view_num = view.split(' ')[0]
    if view_num[-1] == 'M':
        view_num = float(view_num.replace('M', '')) * 1000000
    elif view_num[-1] == 'K':
        view_num = float(view_num.replace('K', '')) * 1000
    else:
        view_num = float(view_num)
    return view_num


# In[19]:


data_cnn['views_num'] = data_cnn['views'].apply(lambda x: extract_view(x))
data_cnn.head()


# In[20]:


data_cnn['month_group'].value_counts()


# In[21]:


avg_view = data_cnn.groupby('month_group').agg({'views': 'count', 'views_num': 'sum'}).\
    reset_index()\
    .rename({'views': 'video_count'}, axis=1)
avg_view['avg_view'] = avg_view['views_num'] / avg_view['video_count']
avg_view


# In[22]:


sns.countplot(data=data_cnn, x = 'month_group')
plt.xlabel('Month')
plt.ylabel('Number of videos')
plt.title('Number of videos each months')
plt.show()


# In[23]:


sns.barplot(data=avg_view, x = 'month_group', y='views_num', errorbar=('ci', False))
plt.xlabel('Month')
plt.ylabel('Total views')
plt.title('Total views each months')
plt.show()


# In[24]:


sns.barplot(data=avg_view, x = 'month_group', y='avg_view', errorbar=('ci', False))
plt.xlabel('Month')
plt.ylabel('Avg views')
plt.title('Avg views each months')
plt.show()


# In[ ]:





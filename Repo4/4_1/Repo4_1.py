#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[5]:


df = pd.read_csv('Uncleaned_Cnn.csv')


# In[6]:


df


# In[7]:


# Splitting "views" column into columns "views" and "views_text"
df[['views','views_text']] = df['views'].str.split(expand=True)
df


# In[8]:


# Removing Column "views_txt"
df.drop("views_text",axis=1,inplace=True)


# In[9]:


# Removing duplicates 'title'
df.drop_duplicates('title',inplace=True)


# In[10]:


# Resetting Index after removing duplicates
x = list(range(0,len(df)))
df.index = x
df


# In[12]:


import humanize

# Convert the 'views' column to numeric values
df['views'] = df['views'].str.replace('K', 'e3').str.replace('M', 'e6').astype(float)

# Sort the DataFrame based on views in descending order
sorted_df = df.sort_values('views', ascending=False)

# Get the top 10 titles, views, and when
top_10_titles = sorted_df[['title', 'views', 'when']].head(10)

# Format the 'views' column using humanize library
top_10_titles['views'] = top_10_titles['views'].apply(lambda x: humanize.intword(x))

# Display the table using pandas DataFrame
top_10_titles


# In[13]:


# Combine all the titles into a single string
all_titles = ' '.join(df['title'])

# Tokenize the titles into individual words
words = word_tokenize(all_titles)

# Remove stopwords from the list of words
stopwords_list = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stopwords_list]

# Join the filtered words back into a single string
filtered_titles = ' '.join(filtered_words)

# Create a WordCloud object
wordcloud = WordCloud(width=1000, height=600, background_color='black').generate(filtered_titles)

# Display the word cloud using matplotlib
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:





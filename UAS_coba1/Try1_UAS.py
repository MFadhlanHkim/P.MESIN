#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


# Membaca dataset
df = pd.read_csv('penyakit_menular.csv')


# In[3]:


# Menampilkan 5 baris pertama dari dataset
print("5 Baris Pertama Dataset:")
print(df.head())


# In[4]:


# Informasi umum tentang dataset
print("\nInformasi Dataset:")
print(df.info())


# In[5]:


# Statistik deskriptif dari dataset
print("\nStatistik Deskriptif:")
print(df.describe(include='all'))


# In[6]:


# Menampilkan nama penyakit yang terdapat dalam dataset
print("\nNama Penyakit dalam Dataset:")
print(df['Nama_Penyakit'].unique())


# In[7]:


# Menghitung jumlah penyakit berdasarkan cara penularan
print("\nJumlah Penyakit Berdasarkan Cara Penularan:")
print(df['Cara_Penularan'].value_counts())


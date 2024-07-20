#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Membaca dataset
df = pd.read_csv('Covid_GPT.csv')


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


# Menampilkan negara yang terdapat dalam dataset
print("\nNegara dalam Dataset:")
print(df['Nama_Negara'].unique())


# In[7]:


# Menghitung total kasus per negara
total_kasus_per_negara = df.groupby('Nama_Negara')['Total_Kasus'].max()
print("\nTotal Kasus per Negara:")
print(total_kasus_per_negara)


# In[8]:


# Visualisasi jumlah kasus baru per hari di Indonesia
indonesia_df = df[df['Nama_Negara'] == 'Indonesia']
plt.figure(figsize=(10, 6))
plt.plot(indonesia_df['Tanggal'], indonesia_df['Jumlah_Kasus_Baru'], marker='o')
plt.title('Jumlah Kasus Baru per Hari di Indonesia')
plt.xlabel('Tanggal')
plt.ylabel('Jumlah Kasus Baru')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[9]:


# Visualisasi perbandingan total kasus antar negara
plt.figure(figsize=(10, 6))
sns.barplot(x=total_kasus_per_negara.index, y=total_kasus_per_negara.values)
plt.title('Total Kasus per Negara')
plt.xlabel('Negara')
plt.ylabel('Total Kasus')
plt.show()


# In[10]:


# Menghitung rasio kematian per negara
df['Rasio_Kematian'] = df['Total_Kematian'] / df['Total_Kasus'] * 100
rasio_kematian_per_negara = df.groupby('Nama_Negara')['Rasio_Kematian'].max()
print("\nRasio Kematian per Negara (%):")
print(rasio_kematian_per_negara)


# In[11]:


# Visualisasi rasio kematian per negara
plt.figure(figsize=(10, 6))
sns.barplot(x=rasio_kematian_per_negara.index, y=rasio_kematian_per_negara.values)
plt.title('Rasio Kematian per Negara (%)')
plt.xlabel('Negara')
plt.ylabel('Rasio Kematian (%)')
plt.show()


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Membaca dataset
df = pd.read_csv('cuaca_harian.csv')


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
print(df.describe())


# In[6]:


# Menampilkan kota yang terdapat dalam dataset
print("\nKota dalam Dataset:")
print(df['Nama_Kota'].unique())


# In[7]:


# Menghitung suhu rata-rata per kota
suhu_rata_rata_per_kota = df.groupby('Nama_Kota')['Suhu_Rata_Rata'].mean()
print("\nSuhu Rata-Rata per Kota:")
print(suhu_rata_rata_per_kota)


# In[8]:


# Visualisasi suhu rata-rata per kota
plt.figure(figsize=(10, 6))
sns.barplot(x=suhu_rata_rata_per_kota.index, y=suhu_rata_rata_per_kota.values)
plt.title('Suhu Rata-Rata per Kota')
plt.xlabel('Kota')
plt.ylabel('Suhu Rata-Rata (°C)')
plt.show()


# In[11]:


# Menghitung kecepatan angin rata-rata per kota
kecepatan_angin_rata_rata_per_kota = df.groupby('Nama_Kota')['Kecepatan_Angin'].mean()
print("\nKecepatan Angin Rata-Rata per Kota:")
print(kecepatan_angin_rata_rata_per_kota)


# In[12]:


# Visualisasi kecepatan angin rata-rata per kota
plt.figure(figsize=(10, 6))
sns.barplot(x=kecepatan_angin_rata_rata_per_kota.index, y=kecepatan_angin_rata_rata_per_kota.values)
plt.title('Kecepatan Angin Rata-Rata per Kota')
plt.xlabel('Kota')
plt.ylabel('Kecepatan Angin (km/h)')
plt.show()


# In[14]:


# Visualisasi curah hujan per kota
curah_hujan_per_kota = df.groupby('Nama_Kota')['Curah_Hujan'].sum()
plt.figure(figsize=(10, 6))
sns.barplot(x=curah_hujan_per_kota.index, y=curah_hujan_per_kota.values)
plt.title('Curah Hujan per Kota')
plt.xlabel('Kota')
plt.ylabel('Curah Hujan (mm)')
plt.show()


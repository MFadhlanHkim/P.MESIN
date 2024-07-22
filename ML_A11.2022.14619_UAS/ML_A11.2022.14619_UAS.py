#!/usr/bin/env python
# coding: utf-8

# ## Judul/Project : Prediksi Diabetes Menggunakan Machine Learning
# ###### Nama : Muhammad Fadhlan Hakim
# ###### NIM  : A11.2022.14619
# ###### KLP  : A11.4419

# #### Ringkasan : 
# 
# 
# Tujuan dari kumpulan data ini adalah untuk memprediksi secara diagnostik apakah seorang pasien menderita diabetes,
# berdasarkan pengukuran diagnostik tertentu yang termasuk dalam kumpulan data.
# 
# Kumpulan data ini berasal dari National Institute of Diabetes and Digestive and Kidney
# Penyakit.

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[4]:


# Import dataset
df = pd.read_csv('diabetes.csv')


# ### Penjelasan dataset:
# 
# Dataset ini berisi informasi medis dari pasien, dengan kolom-kolom sebagai berikut:
# > 1. Pregnancies: Jumlah kehamilan
# > 2. Glucose: Kadar glukosa plasma 2 jam dalam tes toleransi glukosa oral
# > 3. BloodPressure: Tekanan darah diastolik (mm Hg)
# > 4. SkinThickness: Ketebalan lipatan kulit triseps (mm)
# > 5. Insulin: Kadar insulin serum 2 jam (mu U/ml)
# > 6. BMI: Indeks massa tubuh (berat dalam kg/(tinggi dalam m)^2)
# > 7. DiabetesPedigreeFunction: Fungsi riwayat diabetes (probabilitas berdasarkan riwayat keluarga)
# > 8. Age: Usia (tahun)
# > 9. Outcome: Variabel target (1: menderita diabetes, 0: tidak menderita diabetes)
# 

# In[5]:


# Menampilkan beberapa baris pertama dari dataset
print("Tampilan beberapa baris pertama dari dataset:")
print(df.head())


# In[6]:


# Menampilkan informasi dataset
print("\nInformasi dataset:")
print(df.info())


# In[7]:


# Statistik deskriptif dari dataset
print("\nStatistik deskriptif dari dataset:")
print(df.describe())


# In[23]:


# Exploratory Data Analysis (EDA)
# Distribusi kelas
sns.countplot(x='Outcome', data=df)
plt.title('Distribusi Kelas')
plt.show()


# > Tujuan dari plot ini adalah untuk memvisualisasikan distribusi jumlah sampel dalam setiap kelas (0 dan 1) dari variabel target 'Outcome'. 

# In[9]:


# Pairplot untuk melihat hubungan antara fitur-fitur
sns.pairplot(df, hue='Outcome')
plt.show()


# > Korelasi Antar Fitur: Pairplot membantu dalam mengidentifikasi korelasi antara berbagai fitur dalam dataset. Misalnya, kita dapat melihat apakah ada fitur-fitur yang memiliki hubungan linear atau non-linear yang kuat.
# > 
# >Visualisasi Kelas: Dengan menggunakan hue='Outcome', kita dapat melihat bagaimana distribusi titik-titik data dalam scatter plot dipengaruhi oleh kelas 'Outcome'. Ini membantu dalam mengidentifikasi fitur-fitur yang mungkin penting untuk memisahkan kelas-kelas tersebut.
# > 
# >Insight untuk Preprocessing: Jika ada fitur-fitur yang sangat berkorelasi, kita mungkin mempertimbangkan untuk mengurangi dimensi atau melakukan teknik lain seperti Principal Component Analysis (PCA) sebelum membangun model.

# In[10]:


# Heatmap korelasi antar fitur
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap Korelasi Fitur')
plt.show()


# ##### Tujuan: Tujuan dari heatmap korelasi ini adalah untuk memberikan visualisasi hubungan linear antara semua fitur dalam dataset. Ini membantu dalam mengidentifikasi fitur-fitur yang saling berkaitan dan dapat memberikan insight mengenai struktur data.
# >
# >Hasil:
# Heatmap akan menunjukkan matriks korelasi, di mana setiap sel dalam matriks menunjukkan nilai korelasi antara dua fitur.
# Sel dengan warna yang lebih dekat ke merah menunjukkan korelasi positif yang kuat, sementara sel dengan warna yang lebih dekat ke biru menunjukkan korelasi negatif yang kuat.
# Nilai korelasi ditampilkan dalam sel, memberikan informasi kuantitatif tentang hubungan antara fitur-fitur.

# In[11]:


# Preprocessing Data
# Memeriksa nilai yang hilang
print("\nJumlah nilai yang hilang pada setiap kolom:")
print(df.isnull().sum())


# In[12]:


# Mengisi nilai yang hilang dengan median (jika ada)
df.fillna(df.median(), inplace=True)


# In[13]:


# Memisahkan fitur dan label
X = df.drop('Outcome', axis=1)
y = df['Outcome']


# In[14]:


# Pembagian dataset menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


# 3. Proses Learning / Modeling
# Inisialisasi dan pelatihan model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[16]:


# Prediksi menggunakan model
y_pred = model.predict(X_test)


# In[19]:


# 4. Performa Model
# Evaluasi performa model
accuracy = accuracy_score(y_test, y_pred)
print(f'\nAkurasi model: {accuracy * 100:.2f}%')

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[20]:


# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Tidak Diabetes', 'Diabetes'], yticklabels=['Tidak Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# ##### Tujuan: Tujuan dari confusion matrix ini adalah untuk memberikan representasi visual dari kinerja model klasifikasi. Ini membantu dalam memahami jumlah prediksi yang benar dan salah yang dibuat oleh model.
# >Hasil:
# Heatmap akan menunjukkan matriks 2x2 jika kita memiliki dua kelas ('Tidak Diabetes' dan 'Diabetes').
# Setiap sel dalam matriks menunjukkan jumlah sampel yang termasuk dalam kategori tersebut.
# >
# >Matriks ini terdiri dari:
# >
# >True Positives (TP): Jumlah prediksi benar untuk kelas positif (diagonal kanan bawah).
# >
# >True Negatives (TN): Jumlah prediksi benar untuk kelas negatif (diagonal kiri atas).
# >
# >False Positives (FP): Jumlah prediksi salah yang memprediksi positif padahal seharusnya negatif (kanan atas).
# >
# >False Negatives (FN): Jumlah prediksi salah yang memprediksi negatif padahal seharusnya positif (kiri bawah).
# 

# ###### 5. Diskusi Hasil dan Kesimpulan
# 
# Model Random Forest menunjukkan akurasi yang baik dalam memprediksi diabetes, dengan akurasi mencapai 72.08%. 
# Confusion matrix menunjukkan bahwa model mampu membedakan antara pasien yang menderita diabetes dan yang tidak dengan cukup baik.
# 
# Model ini dapat digunakan sebagai alat bantu dalam diagnosis medis untuk mendeteksi diabetes. 
# Namun, penting untuk melakukan validasi lebih lanjut dan mempertimbangkan faktor-faktor lain seperti 
# bias dalam data dan interpretasi klinis sebelum mengimplementasikan model ini dalam praktek medis.
# 
# Kesimpulan: Penggunaan model machine learning seperti Random Forest dapat membantu dalam prediksi diabetes 
# dengan akurasi yang tinggi, namun tetap diperlukan evaluasi lebih lanjut dan validasi klinis.
# 

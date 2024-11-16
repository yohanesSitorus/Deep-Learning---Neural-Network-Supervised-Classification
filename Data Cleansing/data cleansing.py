# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 18:41:53 2024

@author: NITRO 5

Dataset cleansing pada dataset training dan test akan dibagi menjadi 2,
dataset cleaned 1 : missing values kolom popularity diisikan mean, 
kolom instrumentalness dan key dihapus

dataset cleaned 2 : missing values kolom popularity diisikan mean, 
kolom instrumentalness dan key diisi dengan prediksi berdasarkan korelasi 
dengan variabel lain menggunakan KNN Imputation.
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress

# Load data
dfTrain1 = pd.read_csv('train.csv')
dfTest1= pd.read_csv('test.csv')
dfTrain2 = dfTrain1.copy()
dfTest2 = dfTest1.copy()

# ----------------!!!-----------------------
# Hitung jumlah missing value per kolom
missing_values_train = dfTrain1.isna().sum()

# Tampilkan jumlah missing value
print(missing_values_train)

# Jika ingin melihat hanya kolom yang memiliki missing value
missing_values_train = missing_values_train[missing_values_train > 0]
print("Kolom dengan missing values:")
print(missing_values_train)

# ----------------!!!-----------------------
# Hitung jumlah missing value per kolom
missing_values_test = dfTest1.isna().sum()

# Tampilkan jumlah missing value
print(missing_values_test)

# Jika ingin melihat hanya kolom yang memiliki missing value
missing_values_test = missing_values_test[missing_values_test > 0]
print("Kolom dengan missing values:")
print(missing_values_test)

# ----------------!!!----------------------- Data Cleaned 1
# Hapus kolom 'instrumentalness' dan 'key'
dfTrain1 = dfTrain1.drop(["instrumentalness", "key"], axis=1)
dfTest1 = dfTest1.drop(["instrumentalness", "key"], axis=1)

# Isi missing values pada kolom 'popularity' dengan mean
dfTrain1['Popularity'].fillna(dfTrain1['Popularity'].mean(), inplace=True)# Simpan dataset pertama
dfTest1['Popularity'].fillna(dfTest1['Popularity'].mean(), inplace=True)# Simpan dataset pertama

# Simpan dataset pertama
dfTrain1.to_csv('train_cleaned_1.csv', index=False)
dfTest1.to_csv('test_cleaned_1.csv', index=False)

# ----------------!!!----------------------- Data Train Cleaned 2
# Isi missing values pada kolom 'popularity' dengan mean
dfTrain2['Popularity'].fillna(dfTrain2['Popularity'].mean(), inplace=True)
dfTest2['Popularity'].fillna(dfTrain2['Popularity'].mean(), inplace=True)

# Prediksi missing values menggunakan regresi linier manual
def fill_missing_with_regression(df, target_col, predictor_col):
    # Pisahkan data lengkap dan data dengan missing values
    complete_data = df[df[target_col].notnull()]
    missing_data = df[df[target_col].isnull()]
    
    # Hitung regresi linier
    slope, intercept, _, _, _ = linregress(complete_data[predictor_col], complete_data[target_col])
    
    # Prediksi nilai yang hilang
    predicted_values = missing_data[predictor_col] * slope + intercept
    df.loc[df[target_col].isnull(), target_col] = predicted_values

# Isi kolom 'instrumentalness' dengan prediksi berdasarkan 'energy'
fill_missing_with_regression(dfTrain2, 'instrumentalness', 'energy')
fill_missing_with_regression(dfTest2, 'instrumentalness', 'energy')

# Isi kolom 'key' dengan prediksi berdasarkan 'valence'
fill_missing_with_regression(dfTrain2, 'key', 'valence')
fill_missing_with_regression(dfTest2, 'key', 'valence')

# Simpan dataset kedua
dfTrain2.to_csv('train_cleaned_2.csv', index=False)
dfTest2.to_csv('test_cleaned_2.csv', index=False)

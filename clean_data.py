import pandas as pd
import numpy as np
import os

def clean_data():
    raw_path = 'datasets/dataset_rumah99_raw.csv'
    clean_path = 'datasets/dataset_rumah99_clean.csv'
    
    # Load dataset
    df = pd.read_csv(raw_path)
    print(f"Original shape: {df.shape}")
    
    # Drop rows where critical variables are NaN
    df_clean = df.dropna(subset=['Harga_Rupiah', 'Kamar_Tidur', 'Kamar_Mandi', 'Luas_Tanah_m2', 'Luas_Bangunan_m2', 'Jumlah_Lantai', 'Kota'])
    print(f"Shape after dropping NaNs in key columns: {df_clean.shape}")
    
    # Filter out invalid entries: bedrooms, bathrooms, and electricity capacity must be > 0
    df_clean = df_clean[
        (df_clean['Kamar_Tidur'] > 0) & 
        (df_clean['Kamar_Mandi'] > 0) & 
        (df_clean['Daya_Listrik_Watt'] > 0)
    ]
    print(f"Shape after filtering non-zero rooms and electricity: {df_clean.shape}")
    
    # Handle outliers:
    # 1. Harga_Rupiah: Rp 100 Juta s.d. Rp 50 Miliar
    # 2. Luas_Tanah_m2 & Luas_Bangunan_m2: s.d. 2000 m2
    # 3. Kamar_Tidur & Kamar_Mandi: s.d. 15 kamar
    df_clean = df_clean[
        (df_clean['Harga_Rupiah'] >= 100_000_000) & (df_clean['Harga_Rupiah'] <= 50_000_000_000) &
        (df_clean['Luas_Tanah_m2'] <= 2000) & (df_clean['Luas_Bangunan_m2'] <= 2000) &
        (df_clean['Kamar_Tidur'] <= 15) & (df_clean['Kamar_Mandi'] <= 15)
    ]
    print(f"Shape after removing outliers (price, area, rooms): {df_clean.shape}")
    
    # Save the cleaned dataset
    df_clean.to_csv(clean_path, index=False)
    print(f"Cleaned dataset saved successfully to: {clean_path}")
    
if __name__ == "__main__":
    clean_data()

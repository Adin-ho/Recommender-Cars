import pandas as pd
import re

# Load file
df_db = pd.read_csv('app/data/data_mobil.csv')
df_new = pd.read_csv('app/data/mobil_belum_ada.csv')

# Cleaning nama mobil untuk deduplikasi
def clean_name(nama):
    nama = str(nama).strip().lower()
    nama = re.sub(r'[^a-z0-9 ]', '', nama)
    nama = re.sub(r'\s+', ' ', nama)
    return nama.strip()

df_db['clean_nama'] = df_db['Nama Mobil'].apply(clean_name)
df_new['clean_nama'] = df_new['Nama Mobil'].apply(clean_name)

# Gabungkan tanpa duplikat
clean_db = set(df_db['clean_nama'])
df_new_unique = df_new[~df_new['clean_nama'].isin(clean_db)]

df_final = pd.concat([df_db, df_new_unique], ignore_index=True)
df_final = df_final.drop(columns=['clean_nama'])

df_final.to_csv('app/data/data_mobil_final.csv', index=False)
print("File hasil merge disimpan sebagai app/data/data_mobil_final.csv")
print(f"Jumlah baris awal: {len(df_db)} | Baris baru: {len(df_new)} | Total setelah merge: {len(df_final)} | Tambahan unik: {df_new_unique.shape[0]}")

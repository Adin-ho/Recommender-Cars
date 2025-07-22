import pandas as pd

df = pd.read_csv('app/data/data_mobil.csv')
print("Jumlah baris:", df.shape[0])
print("Jumlah kolom:", df.shape[1])
print("\nJumlah Nilai Unik per Kolom:")
for col in df.columns:
    print(f"- {col}: {df[col].nunique()}")
print("\nStatistik (kolom numerik):")
print(df.describe())
import pandas as pd
import re

def clean_car_name(s):
    # Ambil format: nama mobil (tahun)
    match = re.search(r"([a-z0-9 .\-]+)\s*\((\d{4})\)", s.lower())
    if match:
        return f"{match.group(1).strip()} ({match.group(2)})"
    return s.lower().strip()

df = pd.read_csv('evaluation/evaluasi_semua_batch.csv')

def dedup(val):
    if pd.isna(val) or not isinstance(val, str): return ""
    cars = [clean_car_name(x) for x in re.split(r"[;,\n]", val)]
    cars = list(dict.fromkeys([c for c in cars if c]))  # unique, preserve order
    return ";".join(cars)

df['ground_truth'] = df['ground_truth'].apply(dedup)
df.to_csv('evaluasi_semua_batch_dedup.csv', index=False)
print("Selesai. Duplikat ground_truth sudah dibersihkan. File: evaluasi_semua_batch_dedup.csv")

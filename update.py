import pandas as pd

df = pd.read_csv("app/data/data_mobil_final.csv", dtype=str)

def fix_tahun(t):
    if pd.isna(t) or t.strip() == "":
        return 0
    t_str = str(t).strip()
    if '.' in t_str:
        before, _ = t_str.split('.')
        return int(before)
    if t_str.isdigit():
        return int(t_str)
    try:
        return int(float(t_str))
    except:
        return 0

df['Tahun'] = df['Tahun'].apply(fix_tahun)

TAHUN_SAAT_INI = 2025
df['Usia'] = TAHUN_SAAT_INI - df['Tahun'].astype(int)

df.to_csv("app/data/data_mobil_final.csv", index=False)
print("Kolom Usia sudah update, semua data sesuai rumus 2025 - Tahun Mobil!")

import pandas as pd
from difflib import SequenceMatcher
import sys, os, importlib.util
import re

def clean_name(nama):
    nama = str(nama).strip().lower()
    nama = re.sub(r'[^a-z0-9 ]', '', nama)
    nama = re.sub(r'\b(putih|merah|hitam|silver|abu|metalik|km|only|promo|limited|deluxe|std|double blower|special|manual|matic)\b', '', nama)
    nama = re.sub(r'\s+', ' ', nama)
    return nama.strip()

def bersihkan(text):
    if pd.isna(text) or not str(text).strip():
        return '-'
    def clean_nm(nama):
        n = nama.split('(')[0]
        return clean_name(n)
    return ';'.join([clean_nm(x) for x in str(text).split(';') if x.strip()])

def fuzzy_in(a, b_list, threshold=0.7):
    return any(SequenceMatcher(None, a, b).ratio() >= threshold for b in b_list)

def skor_per_baris(gt, pr):
    set_gt = [x.strip() for x in gt.split(';') if x.strip() and gt != '-']
    set_pr = [x.strip() for x in pr.split(';') if x.strip() and pr != '-']
    matched_pred = set()
    tp = 0
    for g in set_gt:
        for i, p in enumerate(set_pr):
            if i in matched_pred:
                continue
            if fuzzy_in(g, [p]):
                tp += 1
                matched_pred.add(i)
                break
    fp = len(set_pr) - tp
    fn = len(set_gt) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return pd.Series([precision, recall, f1])

# Import rule_based
rule_based_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app', 'rule_based.py'))
spec = importlib.util.spec_from_file_location("rule_based", rule_based_path)
rule_based = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rule_based)
jawab = rule_based.jawab

# Baca file evaluasi
import pandas as pd

df = pd.read_csv('hasil_evaluasi_final.csv')
avg_gt = df['ground_truth'].apply(lambda x: len(str(x).split(';')) if pd.notna(x) else 0).mean()
head_n = int(avg_gt * 2)  # head_n minimal 2x rata-rata jawaban
print(f'Rata-rata ground_truth per pertanyaan: {avg_gt:.2f} → set head_n: {head_n}')
print("Generate jawaban model ke kolom prediksi...")
df['prediksi'] = df['pertanyaan'].apply(lambda x: jawab(x) if pd.notna(x) else "-")
df['ground_truth'] = df['ground_truth'].apply(bersihkan)
df['prediksi'] = df['prediksi'].apply(bersihkan)

df[['precision', 'recall', 'f1']] = df.apply(
    lambda row: skor_per_baris(row['ground_truth'], row['prediksi']), axis=1
)
precision_global = df['precision'].mean()
recall_global = df['recall'].mean()
f1_global = df['f1'].mean()
print("\nBaris recall TERTINGGI:")
print(df.sort_values('recall', ascending=False)[['pertanyaan', 'ground_truth', 'prediksi', 'recall', 'f1']].head(5))

print("\nBaris recall TERENDAH:")
print(df.sort_values('recall')[['pertanyaan', 'ground_truth', 'prediksi', 'recall', 'f1']].head(5))

print("\n=== SKOR EVALUASI AKHIR ===")
print(f"Precision: {precision_global:.3f}")
print(f"Recall:    {recall_global:.3f}")
print(f"F1 Score:  {f1_global:.3f}")

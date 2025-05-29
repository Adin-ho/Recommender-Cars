import pandas as pd
import requests
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import ast

CSV_PATH = "evaluation/evaluasi_semua_batch.csv"
API_URL = "http://localhost:8000/jawab"

print("\nEvaluasi dimulai...\n")
df = pd.read_csv(CSV_PATH)

true_labels = []
pred_labels = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    pertanyaan = row["pertanyaan"]
    gt_raw = row["ground_truth"]
    ground_truth = set(map(str.strip, gt_raw.split(";"))) if gt_raw else set()

    try:
        res = requests.get(API_URL, params={"pertanyaan": pertanyaan}, timeout=30)
        if res.status_code != 200:
            raise Exception(f"Status code: {res.status_code}")
        jawaban = res.json()
        pred = set(map(str.strip, jawaban.get("mobil", [])))
    except Exception as e:
        pred = set()

    # Konversi ke format evaluasi
    all_labels = ground_truth.union(pred)
    for label in all_labels:
        true_labels.append(1 if label in ground_truth else 0)
        pred_labels.append(1 if label in pred else 0)

# Hitung metrik
precision = precision_score(true_labels, pred_labels, zero_division=0)
recall = recall_score(true_labels, pred_labels, zero_division=0)
f1 = f1_score(true_labels, pred_labels, zero_division=0)

print("\n📊 Hasil Evaluasi Akhir:")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1:.2f}")

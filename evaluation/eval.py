import pandas as pd
import requests
import time
from sklearn.metrics import precision_score, recall_score, f1_score

# Load data
df = pd.read_csv("evaluasi_semua_batch.csv")

# Simpan hasil
all_precisions, all_recalls, all_f1s = [], [], []

def extract_predicted_mobil(text):
    lines = text.split("\n")
    predicted = []
    for line in lines:
        if line.strip().startswith("1.") or line.strip().startswith("2.") or line.strip().startswith("3.") \
           or line.strip().startswith("4.") or line.strip().startswith("5.") or line.strip()[0].isdigit():
            nama = line.split("**")
            if len(nama) >= 3:
                predicted.append(nama[1].strip())
    return predicted

for idx, row in df.iterrows():
    pertanyaan = row["pertanyaan"]
    gt_raw = row["ground_truth"]

    # Skip jika ground truth kosong
    if not isinstance(gt_raw, str) or gt_raw.strip() == "":
        continue

    ground_truth = set([x.strip().lower() for x in gt_raw.split(";") if x.strip()])

    try:
        res = requests.get("http://localhost:8000/stream", params={"pertanyaan": pertanyaan}, timeout=15)
        content = ""
        for line in res.iter_lines():
            if line:
                if line.startswith(b"data:"):
                    line_decoded = line.decode()[6:]
                    if "\"token\"" in line_decoded:
                        try:
                            token = eval(line_decoded).get("token")
                            content += token
                        except:
                            continue
        pred = extract_predicted_mobil(content)
        pred_set = set([x.strip().lower() for x in pred if x.strip()])

        # hitung metrik
        tp = len(ground_truth & pred_set)
        precision = tp / len(pred_set) if pred_set else 0
        recall = tp / len(ground_truth) if ground_truth else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)

        print(f"✅ {idx+1}. {pertanyaan}\n → P: {precision:.2f}, R: {recall:.2f}, F1: {f1:.2f}\n")

        time.sleep(1.2)  # Hindari overload API

    except Exception as e:
        print(f"❌ {idx+1}. Gagal evaluasi: {pertanyaan}")
        print("   Error:", e)

# Ringkasan
print("\n📊 Ringkasan Evaluasi:")
print(f"🔸 Precision rata-rata: {sum(all_precisions)/len(all_precisions):.3f}")
print(f"🔸 Recall rata-rata:    {sum(all_recalls)/len(all_recalls):.3f}")
print(f"🔸 F1 Score rata-rata:  {sum(all_f1s)/len(all_f1s):.3f}")

import pandas as pd
import requests
import matplotlib.pyplot as plt
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Load dataset evaluasi
df = pd.read_csv("evaluasi_semua_batch.csv")

# Ambil semua mobil unik dari ground truth
all_mobil_unik = set()
for gt in df["ground_truth"].dropna():
    all_mobil_unik.update(x.strip().lower() for x in gt.split(";"))

# ✅ Fungsi ekstraksi yang lebih fleksibel
def extract_predicted_mobil(text):
    lines = text.split("\n")
    predicted = set()

    for line in lines:
        # Tangkap format: **Nama Mobil**
        match = re.search(r"\*\*(.*?)\*\*", line)
        if match:
            predicted.add(match.group(1).strip().lower())
    return predicted

results = []
all_precisions, all_recalls, all_f1s = [], [], []

for idx, row in df.iterrows():
    pertanyaan = row["pertanyaan"]
    gt_raw = row["ground_truth"]

    if not isinstance(gt_raw, str) or not gt_raw.strip():
        continue

    ground_truth = set(x.strip().lower() for x in gt_raw.split(";") if x.strip())

    try:
        res = requests.get("http://localhost:8000/stream", params={"pertanyaan": pertanyaan}, timeout=30)
        content = ""
        for line in res.iter_lines():
            if line and line.startswith(b"data:"):
                token = eval(line.decode()[6:]).get("token")
                if token:
                    content += token

        pred_set = extract_predicted_mobil(content)

        # ✅ Debug output
        print(f"\n🧪 {idx+1}. {pertanyaan}")
        print("📦 Output:", content[:200], "..." if len(content) > 200 else "")
        print("🎯 Prediksi:", pred_set)
        print("🎯 Ground truth:", ground_truth)

        domain = list(all_mobil_unik | ground_truth | pred_set)
        y_true = [1 if x in ground_truth else 0 for x in domain]
        y_pred = [1 if x in pred_set else 0 for x in domain]

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"✅ P: {precision:.2f}, R: {recall:.2f}, F1: {f1:.2f}")

        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)

        results.append({
            "pertanyaan": pertanyaan,
            "ground_truth": ";".join(ground_truth),
            "prediksi": ";".join(pred_set),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3)
        })

    except Exception as e:
        print(f"❌ {idx+1}. Gagal evaluasi: {pertanyaan}")
        print("   Error:", e)

# Simpan hasil ke CSV
pd.DataFrame(results).to_csv("hasil_evaluasi.csv", index=False)

# 📈 Grafik gabungan
plt.figure(figsize=(10, 5))
plt.plot(all_precisions, label="Precision", color="blue")
plt.plot(all_recalls, label="Recall", color="green")
plt.plot(all_f1s, label="F1 Score", color="red")
plt.title("Evaluasi Model per Pertanyaan")
plt.xlabel("Pertanyaan ke-")
plt.ylabel("Skor")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("grafik_evaluasi_gabungan.png")

# 📉 Grafik terpisah
fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
axs[0].plot(all_precisions, color="blue")
axs[0].set_title("Precision per Pertanyaan")
axs[0].set_ylabel("Precision")
axs[0].grid(True)

axs[1].plot(all_recalls, color="green")
axs[1].set_title("Recall per Pertanyaan")
axs[1].set_ylabel("Recall")
axs[1].grid(True)

axs[2].plot(all_f1s, color="red")
axs[2].set_title("F1 Score per Pertanyaan")
axs[2].set_ylabel("F1 Score")
axs[2].set_xlabel("Pertanyaan ke-")
axs[2].grid(True)

plt.tight_layout()
plt.savefig("grafik_evaluasi_terpisah.png")

print("\n✅ Evaluasi selesai dan disimpan:")
print("- hasil_evaluasi.csv")
print("- grafik_evaluasi_gabungan.png")
print("- grafik_evaluasi_terpisah.png")

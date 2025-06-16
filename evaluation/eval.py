import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import argparse
from app.rule_based import jawab as jawab_rule
from app.llm_core import jawab as jawab_llm


def evaluate_row(row, source):
    pertanyaan = row["pertanyaan"]
    jawaban_gt = row["ground_truth"]
    ground_truth = set(map(str.strip, jawaban_gt.lower().split(";")))

    try:
        prediksi = jawab_rule(pertanyaan) if source == "rule_based" else jawab_llm(pertanyaan)
    except Exception as e:
        print(f"[❌ ERROR] {source.upper()} gagal. Pertanyaan: {pertanyaan}")
        return {"pertanyaan": pertanyaan, "ground_truth": jawaban_gt, "prediksi": "", "precision": 0, "recall": 0, "f1": 0}

    predicted = set(map(str.strip, prediksi.lower().split(";")))

    true_positive = len(ground_truth & predicted)
    precision = true_positive / len(predicted) if predicted else 0
    recall = true_positive / len(ground_truth) if ground_truth else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    return {
        "pertanyaan": pertanyaan,
        "ground_truth": jawaban_gt,
        "prediksi": "; ".join(predicted),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["rule_based", "llm"], default="rule_based")
    args = parser.parse_args()

    print(f"\n📊 Evaluasi dimulai menggunakan sumber: {args.source}")

    df = pd.read_csv("evaluation/evaluasi_semua_batch.csv")
    if not {"pertanyaan", "ground_truth"}.issubset(df.columns):
        raise ValueError("File CSV harus punya kolom: pertanyaan, ground_truth")

    hasil = df.apply(lambda row: evaluate_row(row, args.source), axis=1)
    df_hasil = pd.DataFrame(list(hasil))

    avg = df_hasil[["precision", "recall", "f1"]].mean()
    print("\n--- Rata-rata ---")
    print(f"Precision: {avg['precision']:.2f}, Recall: {avg['recall']:.2f}, F1: {avg['f1']:.2f}")

    output_path = "evaluation/hasil_evaluasi.csv"
    df_hasil.to_csv(output_path, index=False)
    print(f"\n✅ Hasil evaluasi disimpan ke: {output_path}")

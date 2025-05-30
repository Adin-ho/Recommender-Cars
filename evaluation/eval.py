import pandas as pd
import requests
import numpy as np
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source", choices=["rule_based", "llm"], default="rule_based", help="Pilih sumber jawaban")
parser.add_argument("--csv", default="evaluation/evaluasi_semua_batch.csv", help="Path ke file CSV")
args = parser.parse_args()

CSV_PATH = args.csv
API_URL_RULE = "http://localhost:8000/jawab"
API_URL_LLM = "http://localhost:8000/jawab_llm"

def clean(s):
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\s+\(\d{4}\)\s+\(\d{4}\)', lambda m: f" ({m.group(1)})", s)
    return s.strip()

def parse_list(s):
    if pd.isna(s) or s is None:
        return set()
    return set(clean(x) for x in re.split(r"[;]", str(s)) if clean(x))

df = pd.read_csv(CSV_PATH).fillna("")

print(f"Evaluasi dimulai menggunakan sumber: {args.source}\n")

all_prec, all_rec, all_f1 = [], [], []

for idx, row in df.iterrows():
    pertanyaan = row['pertanyaan']
    gt_raw = row['ground_truth']
    gt_set = parse_list(gt_raw)

    try:
        api_url = API_URL_RULE if args.source == "rule_based" else API_URL_LLM
        res = requests.get(api_url, params={"pertanyaan": pertanyaan}, timeout=20)

        if args.source == "rule_based":
            pred_raw = res.text.strip()
        else:
            pred_raw = res.json().get("jawaban", "").strip()

        pred_set = parse_list(pred_raw)

        if not pred_set:
            print(f"[{idx+1}] \u26a0\ufe0f Jawaban kosong atau tidak ter-parse: '{pred_raw}'")

        tp = len(gt_set & pred_set)
        precision = tp / len(pred_set) if pred_set else 0
        recall = tp / len(gt_set) if gt_set else 0
        f1 = (2 * precision * recall) / (precision + recall) if precision+recall > 0 else 0

        print(f"[{idx+1}] \u2705 Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
        all_prec.append(precision)
        all_rec.append(recall)
        all_f1.append(f1)
    except Exception as e:
        print(f"[{idx+1}] \u274c Error: {e}")

if all_f1:
    print("\n--- Rata-rata ---")
    print(f"Precision: {np.mean(all_prec):.2f}, Recall: {np.mean(all_rec):.2f}, F1: {np.mean(all_f1):.2f}")
else:
    print("Tidak ada evaluasi yang sukses.")

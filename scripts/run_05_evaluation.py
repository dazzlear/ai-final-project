"""Module 5 — Evaluation summary.

Reads the scores CSV files produced by Modules 3 and 4, then prints AUC
and TPR@5%FPR for every method and every target model.

Produces:
    outputs/evaluation_summary.csv   — one row per (method, model)
"""
import argparse
import csv
import pandas as pd
from pathlib import Path

from src.metrics import compute_auc, tpr_at_fpr

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mink_dir",     required=True)   # reads from DRIVE_03
    p.add_argument("--baseline_dir", required=True)   # reads from DRIVE_04
    p.add_argument("--output_dir",   required=True)   # writes to DRIVE_05
    p.add_argument("--model_keys",   nargs="+", required=True)
    p.add_argument("--lengths",      nargs="+", type=int, required=True)
    p.add_argument("--settings",     nargs="+", default=["original"])
    return p.parse_args()

args = parse_args()
MINK_DIR = Path(args.mink_dir)
BASELINE_DIR = Path(args.baseline_dir)
OUT_DIR = Path(args.output_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_KEYS = args.model_keys
LENGTHS = args.lengths
SETTINGS = args.settings

# Load scores from CSV files (Module 3 and 4 outputs)
data = {}
for setting in SETTINGS:
    for length in LENGTHS:
        for model_key in MODEL_KEYS:
            # Try reading Min-K% from Module 3
            mink_path = MINK_DIR / f"mink_scores_{model_key}_{length}_{setting}.csv"
            baseline_path = BASELINE_DIR / f"baseline_scores_{model_key}_{length}_{setting}.csv"
            
            key = f"{model_key}_{length}_{setting}"
            if not (mink_path.exists() or baseline_path.exists()):
                print(f"WARNING: {key} scores not found — skipping")
                continue
            
            # Merge scores from both files if they exist
            if mink_path.exists():
                mink_df = pd.read_csv(mink_path)
                data[key] = mink_df
            if baseline_path.exists():
                baseline_df = pd.read_csv(baseline_path)
                if key in data:
                    data[key] = data[key].merge(baseline_df, on="label", suffixes=("", "_baseline"))
                else:
                    data[key] = baseline_df

if not data:
    raise FileNotFoundError("No score CSV files found. Run Modules 3 and 4 first.")

# Write evaluation summary
EVAL_OUT = OUT_DIR / "evaluation_summary.csv"
with open(EVAL_OUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["method", "model", "length", "setting", "auc", "tpr_at_5fpr"])
    
    for key, df in data.items():
        parts = key.split("_")
        if len(parts) == 3:
            model_key, length, setting = parts
        else:
            continue
        
        # Assume CSV has 'label' column and method columns
        labels = df["label"].values
        
        # Get all score columns (exclude label)
        for method in df.columns:
            if method == "label":
                continue
            scores = df[method].values
            try:
                auc = compute_auc(scores, labels)
                tpr = tpr_at_fpr(scores, labels)
                writer.writerow([method, model_key, length, setting, f"{auc:.4f}", f"{tpr:.4f}"])
            except Exception as e:
                print(f"Skip {method} for {key}: {e}")

print(f"Saved → {EVAL_OUT}")
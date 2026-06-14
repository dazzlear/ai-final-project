"""Module 05 — Evaluation."""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.metrics import compute_auc, tpr_at_fpr

BASELINE_METHODS = ["PPL", "Zlib", "Lowercase", "Neighbor", "Smaller Ref"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mink_dir",     required=True)
    p.add_argument("--baseline_dir", required=True)
    p.add_argument("--output_dir",   required=True)
    p.add_argument("--model_keys",   nargs="+", required=True)
    p.add_argument("--lengths",      nargs="+", type=int, required=True)
    p.add_argument("--settings",     nargs="+", default=["original"])
    return p.parse_args()

def main():
    args     = parse_args()
    mink_dir = Path(args.mink_dir)
    base_dir = Path(args.baseline_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows      = []
    raw_rows  = []   # ← NEW: stores per-sample scores for exact ROC curves

    for model_key in args.model_keys:
        for length in args.lengths:
            for setting in args.settings:

                mink_path = mink_dir / f"mink_scores_{model_key}_len{length}_{setting}.csv"
                if not mink_path.exists():
                    print(f"WARNING: {mink_path.name} not found — skipping")
                    continue
                df_mink = pd.read_csv(mink_path)

                labels      = df_mink["label"].tolist()
                mink_scores = df_mink["min_k_score"].tolist()

                rows.append({
                    "method":      "Min-K%",
                    "model_key":   model_key,
                    "length":      length,
                    "setting":     setting,
                    "auc":         round(compute_auc(mink_scores, labels), 4),
                    "tpr_at_5fpr": round(tpr_at_fpr(mink_scores, labels), 4),
                })

                # ── NEW: save per-sample scores ──────────────────────────
                for score, label in zip(mink_scores, labels):
                    raw_rows.append({
                        "method":    "Min-K%",
                        "model_key": model_key,
                        "length":    length,
                        "setting":   setting,
                        "score":     score,
                        "label":     label,
                    })
                # ─────────────────────────────────────────────────────────

                base_path = base_dir / f"baselines_{model_key}_len{length}_{setting}.csv"
                if not base_path.exists():
                    print(f"WARNING: {base_path.name} not found — baselines skipped")
                    continue
                df_base = pd.read_csv(base_path)

                for method in BASELINE_METHODS:
                    if method not in df_base.columns:
                        print(f"WARNING: column '{method}' missing in {base_path.name}")
                        continue
                    scores = df_base[method].tolist()
                    if any(s != s for s in scores):
                        print(f"WARNING: NaN in {method} for {model_key} len={length} {setting} — skipping")
                        continue
                    rows.append({
                        "method":      method,
                        "model_key":   model_key,
                        "length":      length,
                        "setting":     setting,
                        "auc":         round(compute_auc(scores, labels), 4),
                        "tpr_at_5fpr": round(tpr_at_fpr(scores, labels), 4),
                    })

                    # ── NEW: baselines raw scores too ────────────────────
                    for score, label in zip(scores, labels):
                        raw_rows.append({
                            "method":    method,
                            "model_key": model_key,
                            "length":    length,
                            "setting":   setting,
                            "score":     score,
                            "label":     label,
                        })
                    # ─────────────────────────────────────────────────────

    if not rows:
        raise FileNotFoundError("No score CSV files found. Run Modules 3 and 4 first.")

    # Write evaluation_summary.csv (unchanged)
    out_path = out_dir / "evaluation_summary.csv"
    fieldnames = ["method", "model_key", "length", "setting", "auc", "tpr_at_5fpr"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved → {out_path}")

    # ── NEW: Write raw_scores.csv ────────────────────────────────────────
    raw_path = out_dir / "raw_scores.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "model_key", "length", "setting", "score", "label"])
        writer.writeheader()
        writer.writerows(raw_rows)
    print(f"Saved → {raw_path}")
    # ─────────────────────────────────────────────────────────────────────

    print(f"Total rows written: {len(rows)}")

    df_eval = pd.DataFrame(rows)
    METHODS = ["Neighbor", "PPL", "Zlib", "Lowercase", "Smaller Ref", "Min-K%"]
    for setting in args.settings:
        print(f"\n{'='*60}\n  Setting: {setting}\n{'='*60}")
        df_s = df_eval[df_eval["setting"] == setting]
        for method in METHODS:
            df_m = df_s[df_s["method"] == method]
            if df_m.empty:
                continue
            print(f"  {method:<14}  avg AUC={df_m['auc'].mean():.3f}  avg TPR@5%={df_m['tpr_at_5fpr'].mean():.3f}")

if __name__ == "__main__":
    main()
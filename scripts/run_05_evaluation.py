"""Module 05 — Evaluation.

Merges Min-K% scores (Module 03) and baseline scores (Module 04),
computes AUC and TPR@5%FPR for every method × model × length × setting,
and writes evaluation_summary.csv.

"""
import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

from src.metrics import compute_auc, tpr_at_fpr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASELINE_METHODS = ["PPL", "Zlib", "Lowercase", "Neighbor", "Smaller Ref"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mink_dir",     required=True,
                   help="DRIVE_03 — folder containing mink_scores_*.csv")
    p.add_argument("--baseline_dir", required=True,
                   help="DRIVE_04 — folder containing baselines_*.csv")
    p.add_argument("--output_dir",   required=True,
                   help="DRIVE_05 — writes evaluation_summary.csv")
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

    rows = []  # one dict per (method, model_key, length, setting)

    for model_key in args.model_keys:
        for length in args.lengths:
            for setting in args.settings:

                # Load Min-K% CSV
                mink_path = mink_dir / f"mink_scores_{model_key}_len{length}_{setting}.csv"
                if not mink_path.exists():
                    print(f"WARNING: {mink_path.name} not found — skipping")
                    continue
                df_mink = pd.read_csv(mink_path)

                labels     = df_mink["label"].tolist()
                mink_scores = df_mink["min_k_score"].tolist()

                # Evaluate Min-K%
                rows.append({
                    "method":    "Min-K%",
                    "model_key": model_key,
                    "length":    length,
                    "setting":   setting,
                    "auc":       round(compute_auc(mink_scores, labels), 4),
                    "tpr_at_5fpr": round(tpr_at_fpr(mink_scores, labels), 4),
                })

                # Load Baseline CSV
                base_path = base_dir / f"baselines_{model_key}_len{length}_{setting}.csv"
                if not base_path.exists():
                    print(f"WARNING: {base_path.name} not found — baselines skipped")
                    continue
                df_base = pd.read_csv(base_path)

                # Evaluate each baseline
                for method in BASELINE_METHODS:
                    if method not in df_base.columns:
                        print(f"WARNING: column '{method}' missing in {base_path.name}")
                        continue
                    scores = df_base[method].tolist()
                    if any(s != s for s in scores):  # NaN check
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

    if not rows:
        raise FileNotFoundError("No score CSV files found. Run Modules 3 and 4 first.")

    # Write evaluation_summary.csv
    out_path = out_dir / "evaluation_summary.csv"
    fieldnames = ["method", "model_key", "length", "setting", "auc", "tpr_at_5fpr"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved → {out_path}")
    print(f"Total rows written: {len(rows)}")

    # Print summary table
    df_eval  = pd.DataFrame(rows)
    METHODS  = ["Neighbor", "PPL", "Zlib", "Lowercase", "Smaller Ref", "Min-K%"]

    for setting in args.settings:
        print(f"\n{'='*60}")
        print(f"  Setting: {setting}")
        print(f"{'='*60}")
        df_s = df_eval[df_eval["setting"] == setting]

        for method in METHODS:
            df_m = df_s[df_s["method"] == method]
            if df_m.empty:
                continue
            avg_auc = df_m["auc"].mean()
            avg_tpr = df_m["tpr_at_5fpr"].mean()
            print(f"  {method:<14}  avg AUC={avg_auc:.3f}  avg TPR@5%={avg_tpr:.3f}")


if __name__ == "__main__":
    main()
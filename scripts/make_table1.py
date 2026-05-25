"""Generate Table 1 — AUC comparison across methods and models."""

import argparse
import csv
import pandas as pd
from pathlib import Path

from src.metrics import compute_auc

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",  required=True)   # reads evaluation_summary.csv
    p.add_argument("--output_dir", required=True)   # writes outputs
    p.add_argument("--model_keys", nargs="+", required=True)
    return p.parse_args()

args = parse_args()
INPUT_DIR = Path(args.input_dir)
OUT_DIR = Path(args.output_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_KEYS = args.model_keys

# Load evaluation summary
eval_path = INPUT_DIR / "evaluation_summary.csv"
if not eval_path.exists():
    raise FileNotFoundError(f"Evaluation summary not found: {eval_path}")

df = pd.read_csv(eval_path)

# Build table: rows=methods, cols=models
rows = {}
for method in df["method"].unique():
    rows[method] = {}
    for model_key in MODEL_KEYS:
        subset = df[(df["method"] == method) & (df["model"] == model_key)]
        if not subset.empty:
            rows[method][model_key] = subset["auc"].mean()

# Per-method averages
for method in rows:
    vals = list(rows[method].values())
    rows[method]["Avg."] = sum(vals) / len(vals) if vals else 0.0

# Find column-wise best
cols = MODEL_KEYS + ["Avg."]
best = {col: max(rows[m].get(col, 0.0) for m in rows) for col in cols}

# Save CSV
csv_path = OUT_DIR / "table1_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["method"] + MODEL_KEYS + ["avg"])
    for method in sorted(rows.keys()):
        row = [method]
        for col in cols:
            val = rows[method].get(col, None)
            row.append(f"{val:.3f}" if val is not None else "N/A")
        writer.writerow(row)
print(f"Saved → {csv_path}")

# Generate LaTeX
col_header = " & ".join(MODEL_KEYS + ["Avg."])
lines = [
    r"\begin{tabular}{l" + "c" * len(cols) + "}",
    r"\toprule",
    f"Method & {col_header} \\\\",
    r"\midrule",
]
for method in sorted(rows.keys()):
    cells = []
    for col in cols:
        val = rows[method].get(col, None)
        if val is None:
            cells.append("N/A")
        else:
            cell = f"{val:.3f}"
            if abs(val - best[col]) < 1e-9:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
    lines.append(method + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]

latex = "\n".join(lines)
tex_path = OUT_DIR / "table1.tex"
tex_path.write_text(latex)
print(f"Saved → {tex_path}")
print("\n" + latex)
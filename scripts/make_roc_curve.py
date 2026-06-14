"""Generate ROC curve figure for Min-K% Prob across target models."""

import argparse
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt

from src.metrics import roc_points, compute_auc

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",  required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_keys", nargs="+", required=True)
    return p.parse_args()

args = parse_args()
INPUT_DIR = Path(args.input_dir)
OUT_DIR = Path(args.output_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_KEYS = args.model_keys

eval_path = INPUT_DIR / "evaluation_summary.csv"
if not eval_path.exists():
    raise FileNotFoundError(f"Evaluation summary not found: {eval_path}")

df = pd.read_csv(eval_path)

mink_data = df[df["method"] == "Min-K%"]

fig, ax = plt.subplots(figsize=(5, 4))

colors = ["C0", "C1", "C2"]
for color, model_key in zip(colors, MODEL_KEYS):
    subset = mink_data[mink_data["model_key"] == model_key]
    if subset.empty:
        print(f"WARNING: No Min-K% data for {model_key}")
        continue

    scores = subset["score"].tolist()   # ← adjust to your actual column name
    labels = subset["label"].tolist()   # ← adjust to your actual column name

    fpr, tpr = roc_points(scores, labels)          # ← actually compute the curve
    auc_val  = compute_auc(scores, labels)          # ← recompute for accuracy

    ax.plot(fpr, tpr, color=color, lw=1.5, label=f"{model_key} (AUC={auc_val:.3f})")

ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1, label="Random (AUC=0.500)")

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Min-K% Prob")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()

out = OUT_DIR / "roc_curve_min_k.png"
fig.savefig(out, dpi=200)
plt.close(fig)
print(f"Saved → {out}")
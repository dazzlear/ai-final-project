"""Generate ROC curve figure for Min-K% Prob across target models.

Reads scores pkl files and saves:
    figures/roc_curve_min_k.png

"""
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

from src.metrics import roc_points, compute_auc

# Configuration and paths
DRIVE     = "/content/drive/MyDrive/ai-final-project"
SCORE_DIR = Path(f"{DRIVE}/outputs/scores")
FIG_OUT   = Path(f"{DRIVE}/figures")
FIG_OUT.mkdir(parents=True, exist_ok=True)

# Target models and visualization colors
MODELS = ["pythia-2.8b", "gpt-neo-1.3B", "opt-1.3b"]
COLORS = ["C0", "C1", "C2"]

# Generate ROC curves for Min-K% Prob across all target models
fig, ax = plt.subplots(figsize=(5, 4))

for model, color in zip(MODELS, COLORS):
    path = SCORE_DIR / f"scores_{model}.pkl"
    if not path.exists():
        print(f"WARNING: {path} not found — skipping {model}")
        continue

    d      = pickle.load(open(path, "rb"))
    sc     = d["scores"]["Min-K%"]
    labels = d["labels"]

    fpr, tpr = roc_points(sc, labels)
    auc      = compute_auc(sc, labels)
    ax.plot(fpr, tpr, color=color, lw=1.5, label=f"{model} (AUC={auc:.3f})")

# Plot random baseline for comparison
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1, label="Random (AUC=0.500)")

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Min-K% Prob (WIKIMIA length-64)")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()

out = FIG_OUT / "roc_curve_min_k.png"
fig.savefig(out, dpi=200)
plt.close(fig)
print(f"Saved → {out}")
"""Read scores_*.pkl files and save roc_curve_min_k.png."""

import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from src.metrics import roc_points, compute_auc

DRIVE     = "/content/drive/MyDrive/ai-final-project"
SCORE_DIR = Path(f"{DRIVE}/outputs/scores")
FIG_OUT   = Path(f"{DRIVE}/figures"); FIG_OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["pythia-2.8b", "gpt-neo-1.3B", "opt-1.3b"]
COLORS = ["C0", "C1", "C2"]

plt.figure(figsize=(5, 4))

for model, color in zip(MODELS, COLORS):
    path = SCORE_DIR / f"scores_{model}.pkl"
    if not path.exists():
        print(f"WARNING: {path} not found — skipping")
        continue

    d      = pickle.load(open(path, "rb"))
    sc     = d["scores"]["Min-K%"]
    labels = d["labels"]

    fpr, tpr = roc_points(sc, labels)
    auc      = compute_auc(sc, labels)
    plt.plot(fpr, tpr, color=color, label=f"{model} (AUC={auc:.2f})")

# diagonal reference line
plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Min-K% Prob")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_OUT / "roc_curve_min_k.png", dpi=200)
plt.close()
print("Saved → figures/roc_curve_min_k.png")
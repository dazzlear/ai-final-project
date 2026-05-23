"""Dry run for make_roc_curve.py and make_table1.py using fake score pickles."""

import pickle
import random
import csv
import numpy as np
from pathlib import Path

random.seed(42)
np.random.seed(42)

from src.metrics import compute_auc, tpr_at_fpr, roc_points

# ── fake score pickles ─────────────────────────────────────────────────────
OUT       = Path("outputs/make_scripts_test")
SCORE_DIR = OUT / "scores"; SCORE_DIR.mkdir(parents=True, exist_ok=True)
FIG_OUT   = OUT / "figures"; FIG_OUT.mkdir(parents=True, exist_ok=True)

MODELS  = ["pythia-2.8b", "gpt-neo-1.3B", "opt-1.3b"]
METHODS = ["Neighbor", "PPL", "Zlib", "Lowercase", "Smaller Ref", "Min-K%"]

# generate fake scores where Min-K% is slightly better than others
for model in MODELS:
    labels = [1]*50 + [0]*50
    scores = {}
    for method in METHODS:
        if method == "Min-K%":
            # slightly better separation
            sc = [random.gauss(0.7, 0.2) if l == 1
                  else random.gauss(0.3, 0.2) for l in labels]
        else:
            sc = [random.gauss(0.6, 0.3) if l == 1
                  else random.gauss(0.4, 0.3) for l in labels]
        scores[method] = sc
    pickle.dump(
        {"scores": scores, "labels": labels},
        open(SCORE_DIR / f"scores_{model}.pkl", "wb")
    )
print("Fake score pickles created.")

# ══════════════════════════════════════════════════════════════════════════
# TEST 1 — make_roc_curve.py logic
# ══════════════════════════════════════════════════════════════════════════
print("\n=== DRY RUN: make_roc_curve.py ===")
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 4))
for model in MODELS:
    d      = pickle.load(open(SCORE_DIR / f"scores_{model}.pkl", "rb"))
    sc     = d["scores"]["Min-K%"]
    labels = d["labels"]
    fpr, tpr = roc_points(sc, labels)
    auc      = compute_auc(sc, labels)
    plt.plot(fpr, tpr, label=f"{model} (AUC={auc:.2f})")

plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Min-K% Prob")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_OUT / "roc_curve_min_k.png", dpi=200)
plt.close()

assert (FIG_OUT / "roc_curve_min_k.png").exists()
print("roc_curve_min_k.png saved OK")

# ══════════════════════════════════════════════════════════════════════════
# TEST 2 — make_table1.py logic + evaluation_summary.csv
# ══════════════════════════════════════════════════════════════════════════
print("\n=== DRY RUN: make_table1.py ===")

rows = {m: {} for m in METHODS}
data = {}
for model in MODELS:
    data[model] = pickle.load(open(SCORE_DIR / f"scores_{model}.pkl", "rb"))
    for method in METHODS:
        rows[method][model] = compute_auc(
            data[model]["scores"][method], data[model]["labels"]
        )

for method in METHODS:
    vals = list(rows[method].values())
    rows[method]["Avg."] = sum(vals) / len(vals)

# verify LaTeX builds without crashing
cols = MODELS + ["Avg."]
best = {col: max(rows[m].get(col, 0.0) for m in METHODS) for col in cols}
lines = [r"\begin{tabular}{lcccc}", r"\toprule"]
for method in METHODS:
    cells = []
    for col in cols:
        val  = rows[method].get(col, 0.0)
        cell = f"{val:.2f}"
        if abs(val - best[col]) < 1e-9:
            cell = r"\textbf{" + cell + "}"
        cells.append(cell)
    lines.append(method + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
latex = "\n".join(lines)
(OUT / "table1.tex").write_text(latex)
print("table1.tex saved OK")

# evaluation_summary.csv
eval_path = OUT / "evaluation_summary.csv"
with open(eval_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["method", "model", "auc", "tpr_at_5fpr"])
    for model in MODELS:
        for method in METHODS:
            sc     = data[model]["scores"][method]
            labels = data[model]["labels"]
            auc    = compute_auc(sc, labels)
            tpr    = tpr_at_fpr(sc, labels)
            writer.writerow([method, model, f"{auc:.4f}", f"{tpr:.4f}"])

assert eval_path.exists()
print("evaluation_summary.csv saved OK")

# verify CSV has correct number of rows
with open(eval_path) as f:
    rows_csv = list(csv.reader(f))
expected = 1 + len(MODELS) * len(METHODS)  # header + data rows
assert len(rows_csv) == expected, f"Expected {expected} rows, got {len(rows_csv)}"
print(f"CSV row count correct ({expected} rows including header)")

print("\nAll make_scripts dry runs passed.")
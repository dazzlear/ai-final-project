"""Read scores_*.pkl files and print table1.tex to stdout."""

import pickle
from pathlib import Path
from src.metrics import compute_auc, tpr_at_fpr

DRIVE     = "/content/drive/MyDrive/ai-final-project"
SCORE_DIR = Path(f"{DRIVE}/outputs/scores")
TABLE_OUT = Path(f"{DRIVE}/outputs/table1.tex")

MODELS  = ["pythia-2.8b", "gpt-neo-1.3B", "opt-1.3b"]
METHODS = ["Neighbor", "PPL", "Zlib", "Lowercase", "Smaller Ref", "Min-K%"]

# ── load all scores ────────────────────────────────────────────────────────
data = {}
for model in MODELS:
    path = SCORE_DIR / f"scores_{model}.pkl"
    if not path.exists():
        print(f"WARNING: {path} not found — skipping")
        continue
    data[model] = pickle.load(open(path, "rb"))

# ── compute AUC per method per model ──────────────────────────────────────
rows = {m: {} for m in METHODS}
for model in MODELS:
    if model not in data:
        continue
    scores = data[model]["scores"]
    labels = data[model]["labels"]
    for method in METHODS:
        if method in scores:
            rows[method][model] = compute_auc(scores[method], labels)

# ── compute average ────────────────────────────────────────────────────────
for method in METHODS:
    vals = list(rows[method].values())
    rows[method]["Avg."] = sum(vals) / len(vals) if vals else 0.0

# ── find best AUC per column for bolding ──────────────────────────────────
cols = MODELS + ["Avg."]
best = {}
for col in cols:
    best[col] = max(rows[m].get(col, 0.0) for m in METHODS)

# ── build LaTeX ───────────────────────────────────────────────────────────
lines = []
lines.append(r"\begin{tabular}{lcccc}")
lines.append(r"\toprule")
lines.append(r"Method & Pythia-2.8B & GPT-Neo-1.3B & OPT-1.3b & Avg. \\")
lines.append(r"\midrule")

for method in METHODS:
    cells = []
    for col in cols:
        val = rows[method].get(col, 0.0)
        cell = f"{val:.2f}"
        if abs(val - best[col]) < 1e-9:
            cell = r"\textbf{" + cell + "}"
        cells.append(cell)
    lines.append(method + " & " + " & ".join(cells) + r" \\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")

latex = "\n".join(lines)
print(latex)
TABLE_OUT.write_text(latex)
print(f"\nSaved → {TABLE_OUT}")
"""Generate Table 1 — AUC comparison across methods and models.

Reads scores pkl files and writes:
    outputs/table1_results.csv   — machine-readable AUC table
    outputs/table1.tex           — LaTeX tabular block (ready to paste)

"""
import csv
import pickle
from pathlib import Path

from src.metrics import compute_auc, tpr_at_fpr

# Configuration and paths
DRIVE     = "/content/drive/MyDrive/ai-final-project"
SCORE_DIR = Path(f"{DRIVE}/outputs/scores")
OUT_DIR   = Path(f"{DRIVE}/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Target models and MIA detection methods
MODELS  = ["pythia-2.8b", "gpt-neo-1.3B", "opt-1.3b"]
METHODS = ["Neighbor", "PPL", "Zlib", "Lowercase", "Smaller Ref", "Min-K%"]

# Load scores from pickle files
data = {}
for model in MODELS:
    path = SCORE_DIR / f"scores_{model}.pkl"
    if path.exists():
        data[model] = pickle.load(open(path, "rb"))
    else:
        print(f"WARNING: {path} not found — skipping {model}")

# Compute AUC for each (method, model) pair
rows = {}          # rows[method][model] = auc value
for method in METHODS:
    rows[method] = {}
    for model in MODELS:
        if model not in data:
            continue
        sc  = data[model]["scores"].get(method, [])
        lbl = data[model]["labels"]
        if sc:
            rows[method][model] = compute_auc(sc, lbl)

# Calculate per-method averages across all models
for method in METHODS:
    vals = list(rows[method].values())
    rows[method]["Avg."] = sum(vals) / len(vals) if vals else 0.0

# Identify best AUC for each column (for bolding in LaTeX output)
cols = MODELS + ["Avg."]
best = {col: max(rows[m].get(col, 0.0) for m in METHODS) for col in cols}

# Save results to CSV file
csv_path = OUT_DIR / "table1_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["method"] + MODELS + ["avg"])
    for method in METHODS:
        row = [method]
        for col in cols:
            val = rows[method].get(col, None)
            row.append(f"{val:.3f}" if val is not None else "N/A")
        writer.writerow(row)
print(f"Saved → {csv_path}")

# Generate and save LaTeX tabular for publication
col_header = " & ".join(["Pythia-2.8B", "GPT-Neo-1.3B", "OPT-1.3b", "Avg."])
lines = [
    r"\begin{tabular}{lcccc}",
    r"\toprule",
    f"Method & {col_header} \\\\",
    r"\midrule",
]
for method in METHODS:
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
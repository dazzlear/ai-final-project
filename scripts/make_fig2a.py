"""Read fig2a_results.pkl and save fig2a.png."""

import argparse
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",  required=True)   # reads fig2a_results.pkl
    p.add_argument("--output_dir", required=True)   # writes figures
    return p.parse_args()

args = parse_args()
INPUT_DIR = Path(args.input_dir)
OUT_DIR = Path(args.output_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Model sizes in billions of parameters
SIZES = {
    "pythia-160m": 0.16,
    "pythia-410m": 0.41,
    "pythia-1.4b": 1.4,
    "pythia-2.8b": 2.8,
}

results = pickle.load(open(INPUT_DIR / "fig2a_results.pkl", "rb"))

plt.figure(figsize=(5, 3.5))
for method in ["PPL", "Neighbor", "Min-K%"]:
    xs = [SIZES[s] for s in results]
    ys = [results[s][method] for s in results]
    plt.plot(xs, ys, marker="o", label=method)

plt.xscale("log")
plt.xlabel("Model size (B params)")
plt.ylabel("AUC")
plt.title("AUC vs. Model Size (WIKIMIA)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig2a.png", dpi=200)
plt.close()
print(f"Saved → {OUT_DIR / 'fig2a.png'}")
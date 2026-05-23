"""Read fig2a_results.pkl and save fig2a.png."""

import pickle
import matplotlib.pyplot as plt
from pathlib import Path

DRIVE   = "/content/drive/MyDrive/ai-final-project"
OUT     = Path(f"{DRIVE}/outputs")
FIG_OUT = Path(f"{DRIVE}/figures"); FIG_OUT.mkdir(parents=True, exist_ok=True)

# model sizes in billions of parameters
SIZES = {
    "pythia-160m": 0.16,
    "pythia-410m": 0.41,
    "pythia-1.4b": 1.4,
    "pythia-2.8b": 2.8,
}

results = pickle.load(open(OUT / "fig2a_results.pkl", "rb"))

plt.figure(figsize=(5, 3.5))
for method in ["PPL", "Neighbor", "Min-K%"]:
    xs = [SIZES[s] for s in results]
    ys = [results[s][method] for s in results]
    plt.plot(xs, ys, marker="o", label=method)

plt.xscale("log")
plt.xlabel("Model size (B params)")
plt.ylabel("AUC")
plt.title("AUC vs. Model Size (WIKIMIA length-64)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_OUT / "fig2a.png", dpi=200)
plt.close()
print("Saved → figures/fig2a.png")
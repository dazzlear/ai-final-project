"""Dry run for run_fig2a.py logic — no GPU needed."""

import pickle
import random
import torch
import pandas as pd
from pathlib import Path

random.seed(42)
torch.manual_seed(42)

from src.baselines import ppl_score, neighbor_score
from src.metrics import compute_auc

# ── mocks ──────────────────────────────────────────────────────────────────
def load_model(name):
    print(f"  [mock] load_model({name})")
    return None, None

def get_token_logprobs(model, tok, text):
    return [-0.5] * len(text.split())

def min_k_prob(logprobs, k=20):
    n = max(1, int(len(logprobs) * k / 100))
    return sum(sorted(logprobs)[:n]) / n

# ── paths ──────────────────────────────────────────────────────────────────
SAMPLE_CSV    = "data/wikimia_length64_sample.csv"
OUT           = Path("outputs/dry_run_test"); OUT.mkdir(parents=True, exist_ok=True)
PYTHIA_SIZES  = ["pythia-160m", "pythia-410m", "pythia-1.4b", "pythia-2.8b"]

# ── test run_fig2a logic ───────────────────────────────────────────────────
print("\n=== DRY RUN: run_fig2a.py ===")

df     = pd.read_csv(SAMPLE_CSV)
texts  = df["text"].tolist()
labels = df["label"].tolist()

fig2a_results = {}
for size in PYTHIA_SIZES:
    print(f"\n  size={size}")
    model, tok = load_model(f"EleutherAI/{size}")
    fn  = lambda t: get_token_logprobs(model, tok, t)
    lps = [fn(t) for t in texts]

    fig2a_results[size] = {
        "PPL":      compute_auc([ppl_score(lp) for lp in lps], labels),
        "Min-K%":   compute_auc([min_k_prob(lp, k=20) for lp in lps], labels),
        "Neighbor": compute_auc([neighbor_score(t, fn) for t in texts], labels),
    }
    print(f"  {fig2a_results[size]}")

pickle.dump(fig2a_results, open(OUT / "fig2a_results.pkl", "wb"))

# ── verify ─────────────────────────────────────────────────────────────────
print("\n=== Verifying pickle ===")
r2a = pickle.load(open(OUT / "fig2a_results.pkl", "rb"))
assert set(r2a.keys()) == set(PYTHIA_SIZES), "missing model sizes"
for size in PYTHIA_SIZES:
    assert set(r2a[size].keys()) == {"PPL", "Min-K%", "Neighbor"}
print("Pickle keys correct.")
print("\nDry run passed — ready for Colab.")
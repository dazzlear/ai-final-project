"""Run the model-size experiment for Figure 2a.

Evaluates PPL, Neighbor, and Min-K% Prob across four Pythia model sizes
on WIKIMIA length-128 and saves AUC results to:
    outputs/fig2a_results.pkl

Run BEFORE make_fig2a.py.
"""
import argparse
import pickle
import torch
from pathlib import Path

from src.data import load_wikimia
from src.models import load_model, get_token_logprobs
from src.baselines import ppl_score, neighbor_score
from src.methods import min_k_prob
from src.metrics import compute_auc

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",  required=True)   # for load_wikimia (not used currently)
    p.add_argument("--output_dir", required=True)   # writes fig2a_results.pkl
    p.add_argument("--length",     type=int, default=128)
    return p.parse_args()

args = parse_args()
OUT_DIR = Path(args.output_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)
LENGTH = args.length

# Pythia model sizes to evaluate (smallest → largest)
PYTHIA_SIZES = [
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
]

# Load WIKIMIA dataset (specific length per paper)
df = load_wikimia(length=LENGTH)
texts, labels = df["text"].tolist(), df["label"].tolist()
print(f"Dataset loaded: {len(texts)} examples (length={LENGTH})")

# Evaluate MIA methods across Pythia model sizes
results = {}   # results[model_name][method] = AUC value

for model_name in PYTHIA_SIZES:
    short = model_name.split("/")[-1]
    print(f"\n=== {short} ===")

    model, tok = load_model(model_name)
    fn = lambda t, m=model, k=tok: get_token_logprobs(t, m, k)

    lps = [fn(t) for t in texts]

    results[short] = {
        "PPL":      compute_auc([ppl_score(lp) for lp in lps], labels),
        "Min-K%":   compute_auc([min_k_prob(lp, k=20) for lp in lps], labels),
        "Neighbor": compute_auc([neighbor_score(t, fn) for t in texts], labels),
    }

    print(f"  {results[short]}")
    del model, tok
    torch.cuda.empty_cache()

# Save results to pickle file
out = OUT_DIR / "fig2a_results.pkl"
pickle.dump(results, open(out, "wb"))
print(f"\nSaved → {out}")
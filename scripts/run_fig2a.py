import pickle
from pathlib import Path
import torch
from datasets import load_dataset
from src.models import load_model, get_token_logprobs
from src.baselines import ppl_score, neighbor_score
from src.methods import min_k_prob
from src.metrics import compute_auc

PYTHIA_SIZES = ["pythia-160m", "pythia-410m", "pythia-1.4b", "pythia-2.8b"]
ds = load_dataset("swj0419/WikiMIA", split="WikiMIA_length64")
texts = [ex["input"] for ex in ds]; labels = [ex["label"] for ex in ds]

results = {}  # results[size][method] = AUC
for size in PYTHIA_SIZES:
    print(f"\n=== {size} ===")
    model, tok = load_model(f"EleutherAI/{size}")
    fn = lambda t: get_token_logprobs(t, model, tok)
    lps = [fn(t) for t in texts]
    results[size] = {
        "PPL":      compute_auc([ppl_score(lp) for lp in lps], labels),
        "Min-K%":   compute_auc([min_k_prob(lp, k=20) for lp in lps], labels),
        "Neighbor": compute_auc([neighbor_score(t, fn) for t in texts], labels),
    }
    print(results[size])
    del model; torch.cuda.empty_cache()

pickle.dump(results, open("outputs/fig2a_results.pkl", "wb"))
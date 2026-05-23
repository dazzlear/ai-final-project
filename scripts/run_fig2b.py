import pickle
import random
import torch
import pandas as pd
from pathlib import Path

# -- reproducibility --
random.seed(42)
torch.manual_seed(42)

from src.baselines import ppl_score, neighbor_score
from src.methods import min_k_prob
from src.models import load_model, get_token_logprobs
from src.metrics import compute_auc

DRIVE     = "/content/drive/MyDrive/ai-final-project"
OUT       = Path(f"{DRIVE}/outputs"); OUT.mkdir(parents=True, exist_ok=True)
LENGTHS   = [32, 64, 128, 256]
MODEL     = "EleutherAI/pythia-2.8b"

results = {}
for L in LENGTHS:
    print(f"\n=== Length {L} ===")
    df     = pd.read_csv(f"{DRIVE}/data/wikimia_length{L}_processed.csv")
    texts  = df["text"].tolist()
    labels = df["label"].tolist()

    model, tok = load_model(MODEL)
    fn = lambda t: get_token_logprobs(model, tok, t)
    lps = [fn(t) for t in texts]

    results[L] = {
        "PPL":      compute_auc([ppl_score(lp) for lp in lps], labels),
        "Min-K%":   compute_auc([min_k_prob(lp, k=20) for lp in lps], labels),
        "Neighbor": compute_auc([neighbor_score(t, fn) for t in texts], labels),
    }
    print(f"  {results[L]}")
    del model; torch.cuda.empty_cache()

pickle.dump(results, open(f"{DRIVE}/outputs/fig2b_results.pkl", "wb"))
print("\nDone. Saved → outputs/fig2b_results.pkl")
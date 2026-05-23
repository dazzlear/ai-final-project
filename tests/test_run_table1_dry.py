"""Dry run of run_table1.py logic using mocks — no GPU, no real models needed."""

import pickle
import random
import torch
import pandas as pd
from pathlib import Path

random.seed(42)
torch.manual_seed(42)

from src.baselines import (
    ppl_score, zlib_score, lowercase_score,
    smaller_ref_score, neighbor_score,
)
from src.metrics import compute_auc, tpr_at_fpr

# ── mock Hanna's functions ─────────────────────────────────────────────────
def load_model(name):
    print(f"  [mock] load_model({name})")
    return None, None

def get_token_logprobs(model, tok, text):
    # return fake log-probs proportional to text length
    return [-0.5] * len(text.split())

def min_k_prob(logprobs, k=20):
    n = max(1, int(len(logprobs) * k / 100))
    return sum(sorted(logprobs)[:n]) / n

# ── use sample CSV instead of Drive ───────────────────────────────────────
DATA_PATH   = "data/wikimia_length64_sample.csv"
LOGPROB_DIR = Path("outputs/logprobs_test"); LOGPROB_DIR.mkdir(parents=True, exist_ok=True)
SCORE_DIR   = Path("outputs/scores_test");   SCORE_DIR.mkdir(parents=True, exist_ok=True)

REF_MODELS = {
    "EleutherAI/pythia-2.8b": "EleutherAI/pythia-70m",
}
MODELS = list(REF_MODELS.keys())

# ── load sample dataset ───────────────────────────────────────────────────
df     = pd.read_csv(DATA_PATH)
texts  = df["text"].tolist()
labels = df["label"].tolist()
print(f"Dataset loaded: {len(texts)} examples")

# ── main loop (same logic as run_table1.py) ───────────────────────────────
for target_name in MODELS:
    short      = target_name.split("/")[-1]
    cache_path = SCORE_DIR / f"scores_{short}.pkl"

    print(f"\n=== {short} ===")

    # target log-probs
    lp_path = LOGPROB_DIR / f"lp_{short}.pkl"
    if lp_path.exists():
        print("  Loading cached log-probs...")
        target_lps = pickle.load(open(lp_path, "rb"))
    else:
        print("  Computing target log-probs...")
        model, tok = load_model(target_name)
        target_lps = [get_token_logprobs(model, tok, t) for t in texts]
        pickle.dump(target_lps, open(lp_path, "wb"))
        print(f"  Saved → {lp_path}")

    # checkpoint setup
    checkpoint_path = SCORE_DIR / f"checkpoint_{short}.pkl"
    if checkpoint_path.exists():
        print("  Found checkpoint — resuming...")
        checkpoint = pickle.load(open(checkpoint_path, "rb"))
        scores    = checkpoint["scores"]
        start_idx = checkpoint["last_idx"] + 1
        print(f"  Resuming from index {start_idx}")
    else:
        scores = {
            "PPL": [], "Zlib": [], "Min-K%": [],
            "Lowercase": [], "Neighbor": [], "Smaller Ref": []
        }
        start_idx = 0

    # score
    model, tok = load_model(target_name)
    fn = lambda t: get_token_logprobs(model, tok, t)

    for i, (lp, text) in enumerate(
        zip(target_lps[start_idx:], texts[start_idx:]), start=start_idx
    ):
        scores["PPL"].append(ppl_score(lp))
        scores["Zlib"].append(zlib_score(lp, text))
        scores["Min-K%"].append(min_k_prob(lp, k=20))
        scores["Lowercase"].append(lowercase_score(text, fn))
        scores["Neighbor"].append(neighbor_score(text, fn, n_neighbors=5))

        # checkpoint every 3 texts (small dataset so use 3 instead of 50)
        if (i + 1) % 3 == 0:
            pickle.dump({"scores": scores, "last_idx": i}, open(checkpoint_path, "wb"))
            print(f"  Checkpoint saved at index {i+1}/{len(texts)}")

    # smaller ref
    ref_model, ref_tok = load_model(REF_MODELS[target_name])
    for i, (lp_t, text) in enumerate(
        zip(target_lps[start_idx:], texts[start_idx:]), start=start_idx
    ):
        lp_r = get_token_logprobs(ref_model, ref_tok, text)
        scores["Smaller Ref"].append(smaller_ref_score(lp_t, lp_r))

    # evaluate
    print(f"\n  Results for {short}:")
    for method, sc in scores.items():
        auc = compute_auc(sc, labels)
        tpr = tpr_at_fpr(sc, labels)
        print(f"    {method:12s}  AUC={auc:.3f}  TPR@5%FPR={tpr:.3f}")

    pickle.dump({"scores": scores, "labels": labels}, open(cache_path, "wb"))
    checkpoint_path.unlink(missing_ok=True)
    print(f"  Saved → {cache_path}")

print("\nDry run complete — logic OK.")
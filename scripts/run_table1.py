import os
import pickle
import random
import torch
import pandas as pd
from pathlib import Path

# -- reproducibility --
random.seed(42)
torch.manual_seed(42)

from src.baselines import (
    ppl_score, zlib_score, lowercase_score,
    smaller_ref_score, neighbor_score,
)
from src.methods import min_k_prob
from src.models import load_model, get_token_logprobs
from src.metrics import compute_auc, tpr_at_fpr

# ── paths ──────────────────────────────────────────────────────────────────
DRIVE       = "/content/drive/MyDrive/ai-final-project"
DATA_PATH   = f"{DRIVE}/data/wikimia_length64_processed.csv"
LOGPROB_DIR = Path(f"{DRIVE}/outputs/logprobs"); LOGPROB_DIR.mkdir(parents=True, exist_ok=True)
SCORE_DIR   = Path(f"{DRIVE}/outputs/scores");   SCORE_DIR.mkdir(parents=True, exist_ok=True)

REF_MODELS = {
    "EleutherAI/pythia-2.8b":  "EleutherAI/pythia-70m",
    "EleutherAI/gpt-neo-1.3B": "EleutherAI/gpt-neo-125m",
    "facebook/opt-1.3b":       "facebook/opt-350m",
}
MODELS = list(REF_MODELS.keys())

# ── load dataset ───────────────────────────────────────────────────────────
df     = pd.read_csv(DATA_PATH)
texts  = df["text"].tolist()
labels = df["label"].tolist()
print(f"Dataset loaded: {len(texts)} examples")

# ── sanity check on 5 examples ─────────────────────────────────────────────
print("\nRunning sanity check on 5 examples...")
model, tok = load_model(MODELS[0])
fn = lambda t: get_token_logprobs(t, model, tok)
for ex in df.head(5).itertuples():
    s = ppl_score(fn(ex.text))
    print(f"  label={ex.label}  ppl={s:+.3f}")
del model; torch.cuda.empty_cache()
print("Sanity check done.\n")

# ── main loop ──────────────────────────────────────────────────────────────
for target_name in MODELS:
    short      = target_name.split("/")[-1]
    cache_path = SCORE_DIR / f"scores_{short}.pkl"
    if cache_path.exists():
        print(f"SKIP {short} — already scored"); continue

    print(f"\n=== {short} ===")

    # -- target log-probs (cache to Drive) --
    lp_path = LOGPROB_DIR / f"lp_{short}.pkl"
    if lp_path.exists():
        print("  Loading cached log-probs...")
        target_lps = pickle.load(open(lp_path, "rb"))
    else:
        print("  Computing target log-probs...")
        model, tok = load_model(target_name)
        target_lps = [get_token_logprobs(t, model, tok) for t in texts]
        pickle.dump(target_lps, open(lp_path, "wb"))
        print(f"  Saved → {lp_path}")
        del model; torch.cuda.empty_cache()

    # -- checkpoint setup --
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

    # -- score target model methods --
    model, tok = load_model(target_name)
    fn = lambda t: get_token_logprobs(t, model, tok)

    for i, (lp, text) in enumerate(
        zip(target_lps[start_idx:], texts[start_idx:]), start=start_idx
    ):
        scores["PPL"].append(ppl_score(lp))
        scores["Zlib"].append(zlib_score(lp, text))
        scores["Min-K%"].append(min_k_prob(lp, k=20))
        scores["Lowercase"].append(lowercase_score(text, fn))
        scores["Neighbor"].append(neighbor_score(text, fn, n_neighbors=5))

        if (i + 1) % 50 == 0:
            pickle.dump({"scores": scores, "last_idx": i}, open(checkpoint_path, "wb"))
            print(f"  Checkpoint saved at index {i+1}/{len(texts)}")

    del model; torch.cuda.empty_cache()

    # -- smaller ref scores --
    ref_model, ref_tok = load_model(REF_MODELS[target_name])
    for i, (lp_t, text) in enumerate(
        zip(target_lps[start_idx:], texts[start_idx:]), start=start_idx
    ):
        lp_r = get_token_logprobs(text, ref_model, ref_tok)
        scores["Smaller Ref"].append(smaller_ref_score(lp_t, lp_r))

        if (i + 1) % 50 == 0:
            pickle.dump({"scores": scores, "last_idx": i}, open(checkpoint_path, "wb"))
            print(f"  Ref checkpoint saved at index {i+1}/{len(texts)}")

    del ref_model; torch.cuda.empty_cache()

    # -- evaluate and save --
    print(f"\n  Results for {short}:")
    for method, sc in scores.items():
        auc = compute_auc(sc, labels)
        tpr = tpr_at_fpr(sc, labels)
        print(f"    {method:12s}  AUC={auc:.3f}  TPR@5%FPR={tpr:.3f}")

    pickle.dump({"scores": scores, "labels": labels}, open(cache_path, "wb"))
    checkpoint_path.unlink(missing_ok=True)
    print(f"  Saved → {cache_path}")

print("\nAll done.")
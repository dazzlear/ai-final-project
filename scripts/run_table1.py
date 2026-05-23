import os
import pickle
import torch
import pandas as pd
from pathlib import Path

from src.baselines import (
    ppl_score, zlib_score, lowercase_score,
    smaller_ref_score, neighbor_score,
)
from src.methods import min_k_prob      # Hanna's
from src.models import load_model, get_token_logprobs  # Hanna's
from src.metrics import compute_auc, tpr_at_fpr

# ── paths ──────────────────────────────────────────────────────────────────
DRIVE      = "/content/drive/MyDrive/ai-final-project"
DATA_PATH  = f"{DRIVE}/data/wikimia_length64_processed.csv"
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
fn = lambda t: get_token_logprobs(model, tok, t)
for ex in df.head(5).itertuples():
    s = ppl_score(fn(ex.text))
    print(f"  label={ex.label}  ppl={s:+.3f}")
del model; torch.cuda.empty_cache()
print("Sanity check done.\n")

# ── main loop ──────────────────────────────────────────────────────────────
for target_name in MODELS:
    short = target_name.split("/")[-1]
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
        fn = lambda t: get_token_logprobs(model, tok, t)
        target_lps = [fn(t) for t in texts]
        pickle.dump(target_lps, open(lp_path, "wb"))
        print(f"  Saved → {lp_path}")
        del model; torch.cuda.empty_cache()

    fn = lambda t: get_token_logprobs(
        *load_model.cache[target_name], t
    ) if False else None  # placeholder — ref model loaded below

    # -- scores --
    scores = {
        "PPL":      [ppl_score(lp) for lp in target_lps],
        "Zlib":     [zlib_score(lp, t) for lp, t in zip(target_lps, texts)],
        "Min-K%":   [min_k_prob(lp, k=20) for lp in target_lps],
    }

    # lowercase needs model reload
    model, tok = load_model(target_name)
    fn = lambda t: get_token_logprobs(model, tok, t)
    scores["Lowercase"] = [lowercase_score(t, fn) for t in texts]
    scores["Neighbor"]  = [neighbor_score(t, fn, n_neighbors=5) for t in texts]
    del model; torch.cuda.empty_cache()

    # smaller ref needs ref model
    ref_name = REF_MODELS[target_name]
    ref_model, ref_tok = load_model(ref_name)
    ref_lps = [get_token_logprobs(ref_model, ref_tok, t) for t in texts]
    scores["Smaller Ref"] = [smaller_ref_score(t, r) for t, r in zip(target_lps, ref_lps)]
    del ref_model; torch.cuda.empty_cache()

    # -- evaluate --
    print(f"\n  Results for {short}:")
    for method, sc in scores.items():
        auc = compute_auc(sc, labels)
        tpr = tpr_at_fpr(sc, labels)
        print(f"    {method:12s}  AUC={auc:.3f}  TPR@5%FPR={tpr:.3f}")

    pickle.dump({"scores": scores, "labels": labels}, open(cache_path, "wb"))
    print(f"  Saved → {cache_path}")

print("\nAll done.")
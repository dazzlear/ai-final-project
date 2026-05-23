"""Driver for Table 1. Saves scores to outputs/scores_<model>_<setting>.pkl.

Run on your Google account in Colab. Estimated wall-clock: ~2 hr.
"""

import pickle
from pathlib import Path
import torch
from datasets import load_dataset

from src.models import load_model, get_token_logprobs
from src.baselines import (
    ppl_score, zlib_score, lowercase_score,
    smaller_ref_score, neighbor_score,
)
from src.methods import min_k_prob  # Hanna's
from src.metrics import compute_auc, tpr_at_fpr

MODELS = ["EleutherAI/pythia-2.8b", "EleutherAI/gpt-neo-1.3B", "facebook/opt-1.3b"]
REF_MODELS = {  # smaller-ref pairs
    "EleutherAI/pythia-2.8b":   "EleutherAI/pythia-70m",
    "EleutherAI/gpt-neo-1.3B":  "EleutherAI/gpt-neo-125m",
    "facebook/opt-1.3b":        "facebook/opt-350m",
}
SETTINGS = ["original", "paraphrase"]  # Dazel's splits

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

for setting in SETTINGS:
    ds = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length64_{setting}")
    texts = [ex["input"] for ex in ds]
    labels = [ex["label"] for ex in ds]

    for target_name in MODELS:
        cache = OUT / f"scores_{target_name.split('/')[-1]}_{setting}.pkl"
        if cache.exists():
            print(f"SKIP {cache} (already done)"); continue

        print(f"\n=== {target_name} / {setting} ===")
        model, tok = load_model(target_name)
        logprob_fn = lambda t: get_token_logprobs(model, tok, t)

        # Cache target log-probs ONCE for all per-text methods
        target_lps = [logprob_fn(t) for t in texts]

        scores = {
            "PPL":       [ppl_score(lp) for lp in target_lps],
            "Zlib":      [zlib_score(lp, t) for lp, t in zip(target_lps, texts)],
            "Min-K%":    [min_k_prob(lp, k=20) for lp in target_lps],
            # Lowercase needs a second pass through the model:
            "Lowercase": [lowercase_score(t, logprob_fn) for t in texts],
            # Neighbor needs ~5 extra passes per text:
            "Neighbor":  [neighbor_score(t, logprob_fn, n_neighbors=5) for t in texts],
        }
        del model; torch.cuda.empty_cache()

        # Smaller-ref needs a second model
        ref_name = REF_MODELS[target_name]
        ref_model, ref_tok = load_model(ref_name)
        ref_lps = [get_token_logprobs(ref_model, ref_tok, t) for t in texts]
        scores["Smaller Ref"] = [
            smaller_ref_score(t, r) for t, r in zip(target_lps, ref_lps)
        ]
        del ref_model; torch.cuda.empty_cache()

        # AUC summary
        for method, sc in scores.items():
            print(f"  {method:12s}  AUC = {compute_auc(sc, labels):.3f}  "
                  f"TPR@5%FPR = {tpr_at_fpr(sc, labels):.3f}")

        pickle.dump({"scores": scores, "labels": labels}, open(cache, "wb"))
        print(f"  saved → {cache}")
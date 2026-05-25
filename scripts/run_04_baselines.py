"""Module 4 — Baseline scoring.

Loads cached per-token log-probabilities (written by Module 3) and computes
all reference-free and reference-based baseline MIA scores:

    PPL          – Loss Attack (mean log-probability)
    Zlib         – Zlib-calibrated perplexity ratio
    Lowercase    – Lowercase-ratio score
    Neighbor     – DetectGPT-style neighbourhood score
    Smaller Ref  – Reference-model calibration

"""
import pickle
import torch
import pandas as pd
from pathlib import Path
import argparse

from src.data import load_wikimia_all
from src.baselines import (
    ppl_score,
    zlib_score,
    lowercase_score,
    smaller_ref_score,
    neighbor_score,
)
from src.models import load_model, get_token_logprobs

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",    required=True)   # reads mink CSVs from Module 03
    p.add_argument("--output_dir",   required=True)   # writes baseline CSVs
    p.add_argument("--models",       nargs="+", required=True)
    p.add_argument("--model_keys",   nargs="+", required=True)
    p.add_argument("--lengths",      nargs="+", type=int, required=True)
    p.add_argument("--settings",     nargs="+", default=["original"])
    p.add_argument("--skip_existing",action="store_true")
    p.add_argument("--smoke",        action="store_true")
    return p.parse_args()

args = parse_args()
DRIVE_03  = args.input_dir
DRIVE_04  = args.output_dir
LENGTHS   = args.lengths
SETTINGS  = args.settings
SMOKE     = args.smoke
MODEL_MAP = dict(zip(args.model_keys, args.models))

# Configuration and paths
LOGPROB_DIR = Path(DRIVE_03) / "logprobs"
SCORE_DIR   = Path(DRIVE_04) / "scores"
SCORE_DIR.mkdir(parents=True, exist_ok=True)

# Mapping of target models to their smaller reference models for calibration
TARGET_MODELS = {
    "EleutherAI/pythia-2.8b":  "EleutherAI/pythia-70m",
    "EleutherAI/gpt-neo-1.3B": "EleutherAI/gpt-neo-125m",
    "facebook/opt-1.3b":       "facebook/opt-350m",
}

# List of baseline MIA scoring methods to compute
BASELINE_METHODS = ["PPL", "Zlib", "Lowercase", "Neighbor", "Smaller Ref"]

# Load dataset (all length splits)
df     = load_wikimia_all()
texts  = df["text"].tolist()
labels = df["label"].tolist()
print(f"Dataset loaded: {len(texts)} examples")

# Compute baseline scores for each target model
for target_name, ref_name in TARGET_MODELS.items():
    short      = target_name.split("/")[-1]
    lp_path    = LOGPROB_DIR / f"lp_{short}_all.pkl"
    score_path = SCORE_DIR / f"scores_{short}_all.pkl"

    if not lp_path.exists():
        print(f"\nSKIP {short} — logprobs not found (run Module 3 first).")
        continue

    # Load existing scores dict (Module 3 may have written Min-K% already)
    if score_path.exists():
        saved  = pickle.load(open(score_path, "rb"))
        scores = saved.get("scores", {})
    else:
        scores = {}

    # Skip if all baseline methods are already computed for this model
    already_done = all(
        m in scores and len(scores[m]) == len(texts)
        for m in BASELINE_METHODS
    )
    if already_done:
        print(f"SKIP {short} — baselines already complete.")
        continue

    print(f"\n=== {short} ===")
    target_lps = pickle.load(open(lp_path, "rb"))

    # Load target model (required for Lowercase and Neighbor baselines)
    print(f"  Loading target model: {target_name}")
    model, tok = load_model(target_name)
    fn = lambda t, m=model, k=tok: get_token_logprobs(t, m, k)

    print("  Computing PPL …")
    scores["PPL"]       = [ppl_score(lp) for lp in target_lps]

    print("  Computing Zlib …")
    scores["Zlib"]      = [zlib_score(lp, t) for lp, t in zip(target_lps, texts)]

    print("  Computing Lowercase …")
    scores["Lowercase"] = [lowercase_score(t, fn) for t in texts]

    print("  Computing Neighbor …")
    scores["Neighbor"]  = [neighbor_score(t, fn) for t in texts]

    del model, tok
    torch.cuda.empty_cache()

    # Load smaller reference model for calibration
    print(f"  Loading reference model: {ref_name}")
    ref_model, ref_tok = load_model(ref_name)

    print("  Computing Smaller Ref …")
    scores["Smaller Ref"] = [
        smaller_ref_score(lp_t, get_token_logprobs(t, ref_model, ref_tok))
        for lp_t, t in zip(target_lps, texts)
    ]

    del ref_model, ref_tok
    torch.cuda.empty_cache()

    # Save scores dictionary (preserves any Min-K% scores from Module 3)
    pickle.dump({"scores": scores, "labels": labels}, open(score_path, "wb"))
    print(f"  Saved → {score_path}")

print("\nModule 4 complete — all baseline scores written.")
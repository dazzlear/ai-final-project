"""Module 4 — Baseline scoring.

Reads texts from Module 01 CSVs, loads cached logprobs from Module 03
(if available) or recomputes them, and writes one baseline score CSV per
model × length × setting combination.

Output schema: text_id, label, length, PPL, Zlib, Lowercase, Neighbor, Smaller Ref

Usage (from repo root):
    PYTHONPATH=. python scripts/run_04_baselines.py \
        --input_dir  <DRIVE_01> \
        --logprob_dir <DRIVE_03>/logprobs \
        --output_dir <DRIVE_04> \
        --models     EleutherAI/pythia-2.8b EleutherAI/gpt-neo-1.3B facebook/opt-1.3b \
        --model_keys pythia2.8b gptneo1.3b opt1.3b \
        --lengths    32 64 128 256 \
        --settings   original \
        --skip_existing
"""
import argparse
import pickle
import torch
import pandas as pd
from pathlib import Path

from src.data import load_wikimia
from src.baselines import (
    ppl_score, zlib_score, lowercase_score,
    smaller_ref_score, neighbor_score,
)
from src.models import load_model, get_token_logprobs

# ── Smaller Ref pairings (fixed by paper Table 1) ─────────────────────────
REF_MODELS = {
    "EleutherAI/pythia-2.8b":  "EleutherAI/pythia-70m",
    "EleutherAI/gpt-neo-1.3B": "EleutherAI/gpt-neo-125m",
    "facebook/opt-1.3b":       "facebook/opt-350m",
}

BASELINE_METHODS = ["PPL", "Zlib", "Lowercase", "Neighbor", "Smaller Ref"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",    required=True,
                   help="DRIVE_01 — reads wikimia_length{N}_processed.csv")
    p.add_argument("--logprob_dir",  default=None,
                   help="DRIVE_03/logprobs — optional logprob cache from Module 03")
    p.add_argument("--output_dir",   required=True,
                   help="DRIVE_04 — writes baselines_{key}_len{N}_{setting}.csv")
    p.add_argument("--models",       nargs="+", required=True)
    p.add_argument("--model_keys",   nargs="+", required=True)
    p.add_argument("--lengths",      nargs="+", type=int, required=True)
    p.add_argument("--settings",     nargs="+", default=["original"])
    p.add_argument("--skip_existing",action="store_true")
    p.add_argument("--smoke",        action="store_true",
                   help="Run on first 10 rows only for validation")
    return p.parse_args()


def load_logprobs(logprob_dir, model_key, length, setting, texts, model, tok):
    """Load cached logprobs if available, otherwise recompute."""
    if logprob_dir is not None:
        cache = Path(logprob_dir) / f"logprobs_{model_key}_len{length}_{setting}.pkl"
        if cache.exists():
            print(f"    Loading cached logprobs: {cache.name}")
            return pickle.load(open(cache, "rb"))

    print(f"    Computing logprobs (no cache found) …")
    return [get_token_logprobs(t, model, tok) for t in texts]


def main():
    args      = parse_args()
    MODEL_MAP = dict(zip(args.model_keys, args.models))
    OUT_DIR   = Path(args.output_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_key, target_name in MODEL_MAP.items():
        ref_name = REF_MODELS.get(target_name)
        if ref_name is None:
            print(f"WARNING: No ref model defined for {target_name} — Smaller Ref will be skipped.")

        for length in args.lengths:
            # ── Load texts for this length ─────────────────────────────────
            csv_path = Path(args.input_dir) / f"wikimia_length{length}_processed.csv"
            if not csv_path.exists():
                print(f"SKIP — {csv_path.name} not found (run Module 01 first).")
                continue

            df     = pd.read_csv(csv_path)
            texts  = df["text"].tolist()
            labels = df["label"].tolist()

            if args.smoke:
                texts  = texts[:10]
                labels = labels[:10]
                df     = df.iloc[:10]

            for setting in args.settings:
                out_csv = OUT_DIR / f"baselines_{model_key}_len{length}_{setting}.csv"

                if args.skip_existing and out_csv.exists():
                    print(f"SKIP — {out_csv.name} already exists.")
                    continue

                print(f"\n=== {model_key} | len={length} | {setting} ===")

                # ── Load target model ──────────────────────────────────────
                print(f"  Loading target model: {target_name}")
                model, tok = load_model(target_name)
                fn = lambda t, m=model, k=tok: get_token_logprobs(t, m, k)

                target_lps = load_logprobs(
                    args.logprob_dir, model_key, length, setting,
                    texts, model, tok,
                )

                print("  Computing PPL …")
                ppl_scores = [ppl_score(lp) for lp in target_lps]

                print("  Computing Zlib …")
                zlib_scores = [zlib_score(lp, t) for lp, t in zip(target_lps, texts)]

                print("  Computing Lowercase …")
                lc_scores = [lowercase_score(t, fn) for t in texts]

                print("  Computing Neighbor …")
                nb_scores = [neighbor_score(t, fn) for t in texts]

                del model, tok
                torch.cuda.empty_cache()

                # ── Smaller Ref ────────────────────────────────────────────
                if ref_name:
                    print(f"  Loading reference model: {ref_name}")
                    ref_model, ref_tok = load_model(ref_name)

                    print("  Computing Smaller Ref …")
                    sr_scores = [
                        smaller_ref_score(lp_t, get_token_logprobs(t, ref_model, ref_tok))
                        for lp_t, t in zip(target_lps, texts)
                    ]
                    del ref_model, ref_tok
                    torch.cuda.empty_cache()
                else:
                    sr_scores = [None] * len(texts)

                # ── Write CSV ──────────────────────────────────────────────
                out_df = pd.DataFrame({
                    "text_id":     df["text_id"].tolist(),
                    "label":       labels,
                    "length":      length,
                    "PPL":         ppl_scores,
                    "Zlib":        zlib_scores,
                    "Lowercase":   lc_scores,
                    "Neighbor":    nb_scores,
                    "Smaller Ref": sr_scores,
                })
                out_df.to_csv(out_csv, index=False)
                print(f"  Saved → {out_csv}")

    print("\nModule 04 complete.")


if __name__ == "__main__":
    main()
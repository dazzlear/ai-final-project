import pickle
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models import get_token_logprobs


def min_k_prob(logprobs: List[float], k: int = 20) -> float:

    # Compute the Min-K% Prob score for a single text.
    if not logprobs:
        return float("nan")

    k_length = max(1, int(len(logprobs) * k / 100))

    # Sort ascending: most negative (most surprising) tokens come first
    sorted_lp = np.sort(logprobs)

    # Take the k_length most surprising tokens
    min_k_lp = sorted_lp[:k_length]

    # Higher (less negative) mean → model less surprised → member
    return float(np.mean(min_k_lp))


def save_logprobs_cache(logprobs_list: List[List[float]], cache_path: str) -> None:
    """Persist a list of per-sample log-prob lists to disk."""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(logprobs_list, f)
    print(f"[save_logprobs_cache] Saved {len(logprobs_list)} samples → {cache_path}")


def load_logprobs_cache(cache_path: str) -> Optional[List[List[float]]]:
    """Load cached log-prob lists from disk. Returns None if file missing."""
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    print(f"[load_logprobs_cache] Loaded {len(data)} samples from {cache_path}")
    return data


def score_dataset(
    df: pd.DataFrame,
    model,
    tokenizer,
    k: int = 20,
    cache_path: str = "outputs/logprobs_wikimia_len64.pkl",
) -> pd.DataFrame:
    
    # Compute Min-K% Prob scores for every row in df.
    texts  = df["text"].tolist()
    labels = df["label"].tolist()

    logprobs_list = load_logprobs_cache(cache_path)

    # Cache length mismatch check — prevents silent corruption if dataset changes
    if logprobs_list is not None and len(logprobs_list) != len(texts):
        print(
            f"  WARNING: Cache has {len(logprobs_list)} entries but dataset has "
            f"{len(texts)}. Discarding cache and recomputing."
        )
        logprobs_list = None

    if logprobs_list is None:
        print(f"[score_dataset] Computing log-probs for {len(texts)} samples ...")
        logprobs_list = []
        for idx, text in enumerate(tqdm(texts, desc="Token log-probs")):
            try:
                lp = get_token_logprobs(text, model, tokenizer)
            except (RuntimeError, ValueError, OverflowError) as e:
                # RuntimeError: CUDA OOM or tensor issues
                # ValueError: tokenizer encoding failures
                # OverflowError: sequence too long for model
                print(f"  Warning: sample {idx} failed ({type(e).__name__}: {e}). Using empty list.")
                lp = []
            logprobs_list.append(lp)
        save_logprobs_cache(logprobs_list, cache_path)

    scores = [min_k_prob(lp, k=k) for lp in logprobs_list]

    df_out = pd.DataFrame({
        "text_id":     range(len(texts)),
        "text":        texts,
        "label":       labels,
        "min_k_score": scores,
    })

    # Sanity check: member avg should be higher (less negative) than non-member avg
    member_avg    = df_out[df_out["label"] == 1]["min_k_score"].mean()
    nonmember_avg = df_out[df_out["label"] == 0]["min_k_score"].mean()
    if member_avg <= nonmember_avg:
        print(
            f"  WARNING: member avg ({member_avg:.4f}) ≤ non-member avg ({nonmember_avg:.4f}). "
            "Min-K% scores may be inverted — check min_k_prob() sign convention."
        )

    return df_out
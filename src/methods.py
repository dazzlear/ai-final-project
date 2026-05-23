import pickle
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models import get_token_logprobs


def min_k_prob(logprobs: List[float], k: int = 20) -> float:
    if not logprobs:
        return float("nan")

    # Number of tokens to use — at least 1
    k_length = max(1, int(len(logprobs) * k / 100))

    # Sort ascending: most negative log-probs (most surprising) come first
    sorted_lp = np.sort(logprobs)

    # Take the k_length most surprising tokens
    min_k_lp = sorted_lp[:k_length]

    # Negate the mean — matches run.py's sign convention:
    # higher score → model more surprised → more likely non-member
    return float(np.mean(min_k_lp))


def save_logprobs_cache(logprobs_list: List[List[float]], cache_path: str) -> None:
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(logprobs_list, f)
    print(f"[save_logprobs_cache] Saved {len(logprobs_list)} samples → {cache_path}")


def load_logprobs_cache(cache_path: str) -> Optional[List[List[float]]]:
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
   
    texts  = df["text"].tolist()
    labels = df["label"].tolist()

    # --- Try cache first ---
    logprobs_list = load_logprobs_cache(cache_path)

    # --- Compute if no cache ---
    if logprobs_list is None:
        print(f"[score_dataset] Computing log-probs for {len(texts)} samples ...")
        logprobs_list = []
        for idx, text in enumerate(tqdm(texts, desc="Token log-probs")):
            try:
                lp = get_token_logprobs(text, model, tokenizer)
            except Exception as e:
                print(f"  Warning: sample {idx} failed ({e}). Skipping.")
                lp = []
            logprobs_list.append(lp)
        save_logprobs_cache(logprobs_list, cache_path)

    # --- Compute Min-K% scores ---
    scores = [min_k_prob(lp, k=k) for lp in logprobs_list]

    return pd.DataFrame({
        "text_id":     range(len(texts)),
        "text":        texts,
        "label":       labels,
        "min_k_score": scores,
    })
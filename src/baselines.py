"""Baseline membership-inference scoring functions.

CONVENTION: every score function returns a float where HIGHER means
MORE LIKELY to be a member of the pretraining set. This matches
sklearn.metrics.roc_auc_score, which treats label=1 as the positive class.
"""

import math
import zlib
from typing import Callable, List
import random
from transformers import pipeline

def ppl_score(logprobs: List[float]) -> float:
    """LOSS-attack score: negative perplexity.

    Perplexity = exp(-mean(log p(x_i))). Members have lower PPL,
    so we return -PPL. Equivalently, mean(logprobs) is monotone in -PPL
    and is what we actually return (cheaper and numerically safer
    for very long texts where exp(...) can overflow).

    Args:
        logprobs: per-token natural-log probabilities for one example.

    Returns:
        Float; higher = more likely member.
    """
    if len(logprobs) == 0:
        raise ValueError("ppl_score got empty logprobs list")
    return sum(logprobs) / len(logprobs)  # = -log(PPL) up to a constant

def zlib_score(logprobs: List[float], text: str) -> float:
    """Zlib-calibrated score (Carlini et al. 2021).

    Original definition: PPL / zlib_entropy. We want higher = member,
    so we return -log(PPL) / zlib_entropy, which is monotonic in -ratio.

    Args:
        logprobs: per-token log-probs from get_token_logprobs(model, tok, text).
        text: the raw string that produced those log-probs.

    Returns:
        Float; higher = more likely member.
    """
    if len(text) == 0:
        raise ValueError("zlib_score got empty text")
    log_ppl = -sum(logprobs) / len(logprobs)  # = log(PPL)
    zlib_len = len(zlib.compress(text.encode("utf-8")))
    return -log_ppl / zlib_len

def lowercase_score(
    text: str,
    logprob_fn: Callable[[str], List[float]],
) -> float:
    """Lowercase-ratio score (Carlini et al. 2021).

    Compute PPL on original text and on text.lower(). Members typically
    have PPL_original much lower than PPL_lower (model memorized the
    casing). Non-members are insensitive to casing.

    Original definition: PPL_orig / PPL_lower (lower = member).
    We return the negative log-ratio so higher = member.

    Args:
        text: raw string.
        logprob_fn: a closure that takes a string and returns per-token
            log-probs under the TARGET model. Inject this so we don't
            couple baselines.py to model loading.
    """
    lp_orig = logprob_fn(text)
    lp_lower = logprob_fn(text.lower())
    log_ppl_orig = -sum(lp_orig) / len(lp_orig)
    log_ppl_lower = -sum(lp_lower) / len(lp_lower)
    return -(log_ppl_orig - log_ppl_lower)  # = -log(PPL_orig / PPL_lower)

def smaller_ref_score(
    logprobs_target: List[float],
    logprobs_ref: List[float],
) -> float:
    """Reference-model calibration (Carlini et al. 2022).

    score = -(log_ppl_target - log_ppl_ref)
    Higher = more likely member.
    """
    log_ppl_target = -sum(logprobs_target) / len(logprobs_target)
    log_ppl_ref = -sum(logprobs_ref) / len(logprobs_ref)
    return -(log_ppl_target - log_ppl_ref)


_mask_filler = None  # lazy global so we only load BERT once


def _get_mask_filler():
    global _mask_filler
    if _mask_filler is None:
        # distilroberta-base is small and fast; bert-base-uncased also fine
        _mask_filler = pipeline(
            "fill-mask",
            model="distilroberta-base",
            top_k=1,
            device=0,  # GPU if available; set to -1 for CPU
        )
    return _mask_filler


def _generate_neighbors(text: str, n: int = 5, mask_frac: float = 0.15) -> List[str]:
    """Generate n perturbed neighbors by BERT mask-filling.

    For each neighbor: pick mask_frac of word positions, replace with
    <mask>, ask DistilRoBERTa for its top fill.
    """
    filler = _get_mask_filler()
    mask_token = filler.tokenizer.mask_token  # "<mask>" for RoBERTa
    words = text.split()
    if len(words) < 4:
        return [text]  # too short to perturb meaningfully
    n_mask = max(1, int(len(words) * mask_frac))

    neighbors = []
    rng = random.Random(42)  # determinism
    for i in range(n):
        idxs = rng.sample(range(len(words)), n_mask)
        masked_words = list(words)
        # Mask one at a time and refill greedily (avoids multi-mask quirks)
        for idx in idxs:
            original = masked_words[idx]
            masked_words[idx] = mask_token
            try:
                fill = filler(" ".join(masked_words))[0]["token_str"].strip()
                masked_words[idx] = fill if fill else original
            except Exception:
                masked_words[idx] = original
        neighbors.append(" ".join(masked_words))
    return neighbors


def neighbor_score(
    text: str,
    logprob_fn: Callable[[str], List[float]],
    n_neighbors: int = 5,
) -> float:
    """DetectGPT-style neighborhood score (Mattern et al. 2023).

    score = log_ppl(neighbor_mean) - log_ppl(text)
    Members sit at a local minimum, so neighbors have HIGHER PPL than
    text → score is positive. Non-members: comparable PPL → score ≈ 0.
    Higher = member (already in the right direction; no negation).
    """
    lp_text = logprob_fn(text)
    log_ppl_text = -sum(lp_text) / len(lp_text)

    neighbors = _generate_neighbors(text, n=n_neighbors)
    log_ppl_neighbors = []
    for n in neighbors:
        lp_n = logprob_fn(n)
        if len(lp_n) == 0:
            continue
        log_ppl_neighbors.append(-sum(lp_n) / len(lp_n))

    if not log_ppl_neighbors:
        return 0.0  # degenerate; treat as uninformative
    return sum(log_ppl_neighbors) / len(log_ppl_neighbors) - log_ppl_text
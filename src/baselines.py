"""Baseline membership-inference scoring functions.
Convention: higher score = more likely member. Matches sklearn.metrics.roc_auc_score.
"""
import random
import zlib
from typing import Callable, List, Optional, Sequence

import torch
from transformers import pipeline

# Simple, reference-free baselines

def ppl_score(logprobs: Sequence[float]) -> float:
    """LOSS-attack score: negative perplexity (mean log-probability)."""
    if len(logprobs) == 0:
        raise ValueError("ppl_score received an empty logprobs list")
    return sum(logprobs) / len(logprobs)


def zlib_score(logprobs: Sequence[float], text: str) -> float:
    """Zlib-calibrated score: -log(PPL) / zlib_len."""
    if len(logprobs) == 0:
        raise ValueError("zlib_score received an empty logprobs list")
    if len(text) == 0:
        raise ValueError("zlib_score received an empty text string")
    # log(PPL) = -mean(log p)
    log_ppl = -sum(logprobs) / len(logprobs)
    zlib_len = len(zlib.compress(text.encode("utf-8")))
    return -log_ppl / zlib_len


def lowercase_score(
    text: str,
    logprob_fn: Callable[[str], List[float]],
) -> float:
    """Lowercase-ratio score: negated log-ratio of original vs. lowercased PPL.
    
    Members memorize casing; non-members are insensitive.
    """
    lp_orig = logprob_fn(text)
    lp_lower = logprob_fn(text.lower())
    log_ppl_orig = -sum(lp_orig) / len(lp_orig)
    log_ppl_lower = -sum(lp_lower) / len(lp_lower)
    return -(log_ppl_orig - log_ppl_lower)


def smaller_ref_score(
    logprobs_target: Sequence[float],
    logprobs_ref: Sequence[float],
) -> float:
    """Reference-model calibration: -(log_ppl_target - log_ppl_ref).
    
    Members have lower PPL under target model.
    """
    if len(logprobs_target) == 0:
        raise ValueError("smaller_ref_score received empty logprobs_target")
    if len(logprobs_ref) == 0:
        raise ValueError("smaller_ref_score received empty logprobs_ref")
    if len(logprobs_target) != len(logprobs_ref):
        raise ValueError(
            f"smaller_ref_score: logprobs_target ({len(logprobs_target)} tokens) "
            f"and logprobs_ref ({len(logprobs_ref)} tokens) differ in length. "
            "Ensure both use the same tokenizer."
        )
    log_ppl_target = -sum(logprobs_target) / len(logprobs_target)
    log_ppl_ref = -sum(logprobs_ref) / len(logprobs_ref)
    return -(log_ppl_target - log_ppl_ref)


# Neighbor / DetectGPT-style baseline

_mask_filler = None


def _get_mask_filler():
    """Load DistilRoBERTa fill-mask pipeline with GPU if available."""
    global _mask_filler
    if _mask_filler is None:
        device = 0 if torch.cuda.is_available() else -1
        _mask_filler = pipeline(
            "fill-mask",
            model="distilroberta-base",
            top_k=1,
            device=device,
        )
    return _mask_filler


def _generate_neighbors(
    text: str,
    n: int = 5,
    mask_frac: float = 0.15,
    seed: Optional[int] = 42,
) -> List[str]:
    """Generate n perturbed neighbors via BERT mask-filling."""
    filler = _get_mask_filler()
    mask_token = filler.tokenizer.mask_token
    words = text.split()
    if len(words) < 4:
        return [text]

    n_mask = max(1, int(len(words) * mask_frac))
    neighbors = []
    rng = random.Random(seed)

    for _ in range(n):
        idxs = rng.sample(range(len(words)), n_mask)
        masked_words = list(words)
        # Mask and refill one word at a time to avoid multi-mask edge cases
        for idx in idxs:
            original = masked_words[idx]
            masked_words[idx] = mask_token
            try:
                fill = filler(" ".join(masked_words))[0]["token_str"].strip()
                masked_words[idx] = fill if fill else original
            except (RuntimeError, KeyError, IndexError):
                # Restore if GPU/model fails or output format unexpected
                masked_words[idx] = original
        neighbors.append(" ".join(masked_words))

    return neighbors


def neighbor_score(
    text: str,
    logprob_fn: Callable[[str], List[float]],
    n_neighbors: int = 5,
) -> float:
    """DetectGPT-style neighborhood score: mean(log_ppl(neighbors)) - log_ppl(text).
    
    Members sit at loss minima; their neighbors have higher PPL → positive score.
    """
    lp_text = logprob_fn(text)
    log_ppl_text = -sum(lp_text) / len(lp_text)

    neighbors = _generate_neighbors(text, n=n_neighbors)
    log_ppl_neighbors = []
    for nbr in neighbors:
        lp_nbr = logprob_fn(nbr)
        if len(lp_nbr) == 0:
            continue
        log_ppl_neighbors.append(-sum(lp_nbr) / len(lp_nbr))

    if not log_ppl_neighbors:
        # All neighbors degenerate; return neutral score instead of raising
        return 0.0

    return sum(log_ppl_neighbors) / len(log_ppl_neighbors) - log_ppl_text
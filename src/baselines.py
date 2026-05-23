"""Baseline membership-inference scoring functions.

CONVENTION: every score function returns a float where HIGHER means
MORE LIKELY to be a member of the pretraining set. This matches
sklearn.metrics.roc_auc_score, which treats label=1 as the positive class.
"""

import math
import zlib
from typing import Callable, List


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
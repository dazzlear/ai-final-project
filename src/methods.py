"""Min-K% Prob detection method — Shi et al. (ICLR 2024), Section 3.

Four-stage pipeline exposed separately for clarity:
  1. tokenize_text() — tokenization (see log_probability_compute_manual.py /
     log_probability_compute_auto.py)
  2. compute_token_logprobs() — token log probabilities (see
     log_probability_compute_manual.py / log_probability_compute_auto.py)
  3. select_min_k_tokens() — select bottom K%
  4. min_k_prob() — average log likelihood

Two interchangeable implementations of stages 1-2 are available:
  - "manual": log_probability_compute_manual.py — softmax/log/gather done
    with plain Python loops and `math`, for demonstrating the underlying
    computation (smoke tests / verification only, much slower).
  - "auto": log_probability_compute_auto.py — uses
    torch.nn.functional.log_softmax / Tensor.gather(), used for full
    dataset runs.

Select which one to use via the `implementation` argument on score_texts().
"""

import math
import torch
from typing import List, Tuple, Dict, Optional

from . import log_probability_compute_manual as manual_impl
from . import log_probability_compute_auto as auto_impl


_IMPLEMENTATIONS = {
    "manual": manual_impl,
    "auto": auto_impl,
}


def select_min_k_tokens(
    token_logprobs: List[float],
    k: int = 20,
) -> Tuple[List[int], List[float], List[int]]:
    """Select k% of tokens with lowest log probability (outlier tokens).

    Returns: (selected_indices, selected_logprobs, rank_order)
    """
    if not token_logprobs:
        return [], [], []

    # Calculate number of tokens to select
    n_select   = max(1, math.ceil(len(token_logprobs) * k / 100))
    rank_order = sorted(range(len(token_logprobs)), key=lambda i: token_logprobs[i])

    selected_indices  = rank_order[:n_select]
    selected_logprobs = [token_logprobs[i] for i in selected_indices]

    return selected_indices, selected_logprobs, rank_order


def min_k_prob(
    token_logprobs: List[float],
    k: int = 20,
) -> float:
    """Compute Min-K% Prob score (Eq. 1): average log prob of bottom-k% tokens.

    Higher (less negative) = likely member; lower (more negative) = likely non-member.

    Uses plain sum()/len() instead of numpy.mean() to avoid an external
    library dependency for this computation.
    """
    if not token_logprobs:
        return 0.0

    _, selected_logprobs, _ = select_min_k_tokens(token_logprobs, k=k)
    return sum(selected_logprobs) / len(selected_logprobs)


def score_texts(
    texts: List[str],
    model,
    tokenizer,
    k: int = 20,
    device: Optional[str] = None,
    show_progress: bool = True,
    implementation: str = "auto",
) -> List[Dict]:
    """Run full Min-K% Prob pipeline on list of texts, return scores and metadata.

    Args:
        implementation: which log-probability implementation to use.
            "auto"   — torch-optimized (log_probability_compute_auto.py),
                       used for full dataset runs.
            "manual" — plain-Python (log_probability_compute_manual.py),
                       used for smoke tests / verification only, since it
                       is significantly slower.
    """
    if implementation not in _IMPLEMENTATIONS:
        raise ValueError(
            f"Unknown implementation {implementation!r}. "
            f"Must be one of {sorted(_IMPLEMENTATIONS)}."
        )

    compute_token_logprobs = _IMPLEMENTATIONS[implementation].compute_token_logprobs

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = []
    total   = len(texts)

    for i, text in enumerate(texts, 1):
        if show_progress and i % 50 == 0:
            print(f'  scored {i:>4} / {total}')

        lp    = compute_token_logprobs(text, model, tokenizer, device=device)
        score = min_k_prob(lp, k=k)
        _, sel, _ = select_min_k_tokens(lp, k=k)

        results.append({
            'min_k_score': score,
            'n_tokens'   : len(lp),
            'n_selected' : len(sel),
        })

    if show_progress:
        print(f'  scored {total:>4} / {total}  ✓')

    return results
"""Min-K% Prob detection method — Shi et al. (ICLR 2024), Section 3.

Four-stage pipeline exposed separately for clarity:
  1. tokenize_text() — tokenization
  2. compute_token_logprobs() — token log probabilities
  3. select_min_k_tokens() — select bottom K%
  4. min_k_prob() — average log likelihood
"""

import math
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional


def tokenize_text(
    text: str,
    tokenizer,
    max_length: int = 1024,
) -> Dict:
    """Tokenize text. Returns dict with input_ids, tokens, n_tokens, was_truncated."""
    enc = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc['input_ids'][0].tolist()

    tokens = [
        tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        for tid in input_ids
    ]

    # Check if full text exceeds max_length
    full_len = tokenizer(text, return_tensors='pt')['input_ids'].shape[1]
    was_truncated = full_len > max_length

    return {
        'input_ids'    : input_ids,
        'tokens'       : tokens,
        'n_tokens'     : len(input_ids),
        'was_truncated': was_truncated,
    }


def compute_token_logprobs(
    text: str,
    model,
    tokenizer,
    device: Optional[str] = None,
    max_length: int = 1024,
) -> List[float]:
    """Run forward pass, return per-token log probabilities (N-1 values for N tokens).
    
    Computes: log p(x2|x1), log p(x3|x1,x2), ..., log p(xN|x1,...,xN-1)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    enc = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=max_length,
    ).to(device)

    input_ids = enc['input_ids']

    with torch.no_grad():
        outputs = model(**enc)
        logits  = outputs.logits

    # Shift: predict x_i from x_<i
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(
        2, shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    return token_log_probs[0].tolist()


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
    """
    if not token_logprobs:
        return 0.0

    _, selected_logprobs, _ = select_min_k_tokens(token_logprobs, k=k)
    return float(np.mean(selected_logprobs))


def score_texts(
    texts: List[str],
    model,
    tokenizer,
    k: int = 20,
    device: Optional[str] = None,
    show_progress: bool = True,
) -> List[Dict]:
    """Run full Min-K% Prob pipeline on list of texts, return scores and metadata."""
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
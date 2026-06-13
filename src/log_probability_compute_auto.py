"""Tokenization and token log-probability computation for Min-K% Prob.

  1. tokenize_text() — tokenization
  2. compute_token_logprobs() — token log probabilities

These functions handle everything that requires the model and tokenizer
directly. methods.py imports from this module for the scoring stages.
"""

from typing import List, Dict, Optional

import torch


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
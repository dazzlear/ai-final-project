"""Tokenization and token log-probability computation for Min-K% Prob.

  1. tokenize_text() — tokenization
  2. compute_token_logprobs() — token log probabilities

torch is used only to run the model's forward pass and retrieve raw
logits (this is unavoidable since the model itself is a torch model).
Everything after that — softmax, log, and selecting the probability of
the actual next token ("gather") — is computed manually with plain
Python loops and the `math` module, without using
torch.nn.functional.log_softmax or Tensor.gather().

methods.py imports from this module for the scoring stages.
"""

import math
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


def _log_softmax_row(logits_row: List[float]) -> List[float]:
    """Compute log-softmax of a single row of logits using plain math.

    log_softmax(x_i) = x_i - max(x) - log(sum_j exp(x_j - max(x)))

    The max-subtraction is for numerical stability (avoids exp() overflow
    on large logits), same reasoning used internally by
    torch.nn.functional.log_softmax — but implemented here manually.
    """
    max_val = max(logits_row)
    shifted = [x - max_val for x in logits_row]
    sum_exp = sum(math.exp(x) for x in shifted)
    log_sum_exp = math.log(sum_exp)
    return [x - log_sum_exp for x in shifted]


def compute_token_logprobs(
    text: str,
    model,
    tokenizer,
    device: Optional[str] = None,
    max_length: int = 1024,
) -> List[float]:
    """Run forward pass, return per-token log probabilities (N-1 values for N tokens).

    Computes: log p(x2|x1), log p(x3|x1,x2), ..., log p(xN|x1,...,xN-1)

    The forward pass (torch model call) is unavoidable, but the
    log-softmax and "gather" of each target token's log-probability are
    done manually below, without torch.nn.functional.log_softmax or
    Tensor.gather().
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
        logits  = outputs.logits  # shape: (1, seq_len, vocab_size)

    # Move to plain Python lists so all further computation avoids
    # torch tensor operations entirely.
    logits_list = logits[0].tolist()   # seq_len rows, each vocab_size long
    ids_list    = input_ids[0].tolist()  # seq_len token ids

    token_log_probs = []
    n_tokens = len(ids_list)

    for i in range(n_tokens - 1):
        # logits at position i predict the token at position i+1
        row_log_probs = _log_softmax_row(logits_list[i])
        target_token_id = ids_list[i + 1]
        token_log_probs.append(row_log_probs[target_token_id])

    return token_log_probs
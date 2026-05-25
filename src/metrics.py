"""Evaluation metrics for membership-inference scoring.

Convention: higher score = more likely member.
"""

from typing import Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


# Internal helpers

def _validate_inputs(
    scores: Sequence[float],
    labels: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert and validate scores/labels, centralize checks."""
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)

    if scores_arr.shape[0] != labels_arr.shape[0]:
        raise ValueError(
            f"Length mismatch: {len(scores_arr)} scores vs {len(labels_arr)} labels."
        )

    if not np.isfinite(scores_arr).all():
        raise ValueError("scores contain NaN or inf values.")

    unique = np.unique(labels_arr)
    # Labels must only contain 0 and 1
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(
            f"Labels must contain only 0 and 1; got values {unique.tolist()}."
        )
    # Both classes required for AUC computation
    if len(unique) < 2:
        raise ValueError(
            "Both classes (0 and 1) must be present; got only "
            f"{unique.tolist()}. Cannot compute AUC on a single-class dataset."
        )

    return scores_arr, labels_arr


# Public metrics

def compute_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Compute AUC-ROC (labels: 1=member, 0=non-member)."""
    scores_arr, labels_arr = _validate_inputs(scores, labels)
    return float(roc_auc_score(labels_arr, scores_arr))


def tpr_at_fpr(
    scores: Sequence[float],
    labels: Sequence[int],
    fpr: float = 0.05,
) -> float:
    """True-positive rate at target false-positive rate (e.g., TPR@5%FPR)."""
    if not 0.0 <= fpr <= 1.0:
        raise ValueError(f"fpr must be in [0, 1]; got {fpr}.")

    scores_arr, labels_arr = _validate_inputs(scores, labels)
    fprs, tprs, _ = roc_curve(labels_arr, scores_arr)
    return float(np.interp(fpr, fprs, tprs))


def roc_points(
    scores: Sequence[float],
    labels: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fpr_array, tpr_array) for plotting ROC curve."""
    scores_arr, labels_arr = _validate_inputs(scores, labels)
    fpr, tpr, _ = roc_curve(labels_arr, scores_arr)
    return fpr, tpr
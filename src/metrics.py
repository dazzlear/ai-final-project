"""Evaluation metrics for membership inference.

Convention: higher score = more likely member. Matches sklearn.metrics.roc_auc_score.
"""
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Compute ROC-AUC score (higher score = more likely member).

    Args:
        scores: MIA scores (higher = more likely member).
        labels: Binary labels (1 = member, 0 = non-member).
    """
    return roc_auc_score(labels, scores)


def tpr_at_fpr(
    scores: Sequence[float],
    labels: Sequence[int],
    fpr_threshold: float = 0.05,
) -> float:
    """Compute True Positive Rate at a fixed False Positive Rate.

    Args:
        scores: MIA scores (higher = more likely member).
        labels: Binary labels (1 = member, 0 = non-member).
        fpr_threshold: FPR target (default 0.05 for TPR@5%FPR).
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(fpr, fpr_threshold, side="right") - 1
    idx = max(0, min(idx, len(tpr) - 1))
    return float(tpr[idx])


def roc_points(
    scores: Sequence[float],
    labels: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fpr, tpr) arrays for ROC curve plotting.

    Args:
        scores: MIA scores (higher = more likely member).
        labels: Binary labels (1 = member, 0 = non-member).
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    return fpr, tpr
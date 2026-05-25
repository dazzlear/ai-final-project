"""Evaluation metrics for membership-inference scoring."""

from typing import Sequence
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUC-ROC. Labels: 1 = member, 0 = non-member.

    Scores must be higher-is-member (our convention from baselines.py).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError(f"Labels must be 0/1; got {set(labels.tolist())}")
    return float(roc_auc_score(labels, scores))


def tpr_at_fpr(scores: Sequence[float], labels: Sequence[int], fpr: float = 0.05) -> float:
    """True-positive rate at a target false-positive rate.

    Used in Appendix Table 6 of the paper (TPR@5%FPR).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    fprs, tprs, _ = roc_curve(labels, scores)
    # interpolate to the exact target FPR
    return float(np.interp(fpr, fprs, tprs))


def roc_points(scores, labels):
    """Return (fpr_array, tpr_array) for plotting."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    fpr, tpr, _ = roc_curve(labels, scores)
    return fpr, tpr
"""evaluate.py — re-exports all scoring and metric functions.

Per the project plan, this is Jianna's Day 1 deliverable.
Actual implementations live in baselines.py and metrics.py.
"""

from src.baselines import ppl_score, zlib_score
from src.metrics import compute_auc, tpr_at_fpr

__all__ = ["ppl_score", "zlib_score", "compute_auc", "tpr_at_fpr"]
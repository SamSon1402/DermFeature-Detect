"""Classification metrics for the malignancy head and the attribute heads.

Implemented without scikit-learn so the package stays light and the maths is
auditable. Includes the two clinically important extras the demo surfaces:
expected calibration error (ECE) and selective risk / coverage for abstention.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x).ravel().astype(np.float64)


def auroc(scores, labels) -> float:
    """Area under ROC via the Mann-Whitney U statistic (handles ties)."""
    s, y = _np(scores), _np(labels)
    pos, neg = s[y > 0.5], s[y <= 0.5]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts
    avg = (start + cum + 1) / 2.0
    ranks = avg[inv]
    sum_pos = ranks[y > 0.5].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def accuracy(scores, labels, thr: float = 0.5) -> float:
    s, y = _np(scores), _np(labels)
    return float(((s > thr).astype(np.float64) == y).mean())


def sensitivity_specificity(scores, labels, thr: float = 0.5):
    s, y = _np(scores), _np(labels)
    pred = (s > thr).astype(np.float64)
    tp = ((pred == 1) & (y == 1)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return float(sens), float(spec)


def expected_calibration_error(probs, labels, n_bins: int = 15) -> float:
    """ECE: average gap between confidence and accuracy across confidence bins."""
    p, y = _np(probs), _np(labels)
    conf = np.where(p >= 0.5, p, 1 - p)
    correct = ((p >= 0.5).astype(np.float64) == y).astype(np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(p)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def reliability_curve(probs, labels, n_bins: int = 10):
    """Return (confidence, accuracy, weight) per bin for reliability plots."""
    p, y = _np(probs), _np(labels)
    conf = np.where(p >= 0.5, p, 1 - p)
    correct = ((p >= 0.5).astype(np.float64) == y).astype(np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.sum():
            out.append((float(conf[mask].mean()), float(correct[mask].mean()), int(mask.sum())))
    return out


def selective_risk(probs, labels, coverage: float) -> Dict[str, float]:
    """Error rate over the most-confident ``coverage`` fraction of predictions.

    This is what 'abstain to a clinician when unsure' buys you: at lower
    coverage the model only answers where it is confident, and the error on
    those answers should drop.
    """
    p, y = _np(probs), _np(labels)
    conf = np.where(p >= 0.5, p, 1 - p)
    wrong = ((p >= 0.5).astype(np.float64) != y).astype(np.float64)
    order = np.argsort(-conf)
    k = max(1, int(round(coverage * len(p))))
    kept = order[:k]
    return {"coverage": k / len(p), "risk": float(wrong[kept].mean())}


def malignancy_report(probs, labels, thr: float = 0.5) -> Dict[str, float]:
    sens, spec = sensitivity_specificity(probs, labels, thr)
    return {
        "auroc": auroc(probs, labels),
        "accuracy": accuracy(probs, labels, thr),
        "sensitivity": sens,
        "specificity": spec,
        "ece": expected_calibration_error(probs, labels),
    }

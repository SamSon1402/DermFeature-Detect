"""Selective prediction and clinical triage.

Turns a calibrated malignancy probability into one of three actions and decides
when to abstain (defer to a clinician) based on predictive uncertainty. Lower
coverage at a confidence threshold should trade off against lower error on the
cases the model does answer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def predictive_entropy(p: float) -> float:
    """Binary entropy in bits; 1.0 at p=0.5 (max uncertainty), 0 at the extremes."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


@dataclass
class Triage:
    decision: str        # "benign" | "monitor" | "refer"
    abstain: bool        # defer to a clinician?
    probability: float
    entropy: float
    recommendation: str


def triage(prob: float, low: float = 0.33, high: float = 0.66,
           abstain_entropy: float = 0.9) -> Triage:
    h = predictive_entropy(prob)
    abstain = h > abstain_entropy
    if prob < low:
        decision, rec = "benign", "Routine interval scan; no referral indicated."
    elif prob < high:
        decision, rec = "monitor", "Short-interval re-scan; flag colour pattern for review."
    else:
        decision, rec = "refer", "Refer to a dermatologist; biopsy decision is the clinician's."
    if abstain:
        rec = "Low confidence - defer to a clinician. " + rec
    return Triage(decision, abstain, float(prob), float(h), rec)

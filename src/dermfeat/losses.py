"""Losses for multi-task lesion classification.

The malignant class is rare, so the malignancy head uses focal loss (optionally
with a positive-class weight) to stop the easy benign majority from dominating
the gradient. The attribute heads use plain BCE. :class:`MultiTaskLoss` combines
them with configurable weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: float | None = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = None if self.pos_weight is None else torch.tensor(self.pos_weight, device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pw)
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        return ((1 - p_t) ** self.gamma * bce).mean()


class MultiTaskLoss(nn.Module):
    def __init__(self, w_malignancy: float = 1.0, w_attributes: float = 0.5,
                 focal_gamma: float = 2.0, pos_weight: float | None = None):
        super().__init__()
        self.w_mal = w_malignancy
        self.w_attr = w_attributes
        self.mal_loss = FocalLoss(focal_gamma, pos_weight)

    def forward(self, outputs: dict, targets: dict) -> torch.Tensor:
        loss = self.w_mal * self.mal_loss(outputs["malignancy"], targets["malignancy"])
        if "attributes" in outputs and self.w_attr > 0:
            loss = loss + self.w_attr * F.binary_cross_entropy_with_logits(
                outputs["attributes"], targets["attributes"])
        return loss


def build_loss(cfg) -> nn.Module:
    return MultiTaskLoss(
        w_malignancy=cfg.w_malignancy,
        w_attributes=cfg.w_attributes,
        focal_gamma=cfg.focal_gamma,
        pos_weight=cfg.pos_weight,
    )

"""Multi-head classifier.

A shared backbone feeds one head per clinical attribute (A/B/C/D) plus a
malignancy head. Two malignancy modes:

* ``direct``  - malignancy is its own linear head on the pooled features.
* ``fused``   - malignancy is a learned linear combination of the attribute
                logits, so the prediction is an interpretable function of the
                clinical signs (mirrors the ABCDE -> risk fusion in the demo).

Note on attributes: A/B/C/D are derivable from a single image. The fifth
classic sign, *Evolving*, is temporal and is handled by the longitudinal
tracker (project 3), so it is intentionally not a single-image head here.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .backbone import build_backbone


class MultiHeadModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        d = self.backbone.feature_dim
        self.attributes: List[str] = list(cfg.attributes)
        self.dropout = nn.Dropout(cfg.dropout)
        self.attr_head = nn.Linear(d, len(self.attributes))
        self.mode = cfg.malignancy_mode
        if self.mode == "fused":
            self.fuse = nn.Linear(len(self.attributes), 1)
        else:
            self.mal_head = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.dropout(self.backbone(x))
        attr_logits = self.attr_head(feat)
        if self.mode == "fused":
            mal_logits = self.fuse(attr_logits)
        else:
            mal_logits = self.mal_head(feat)
        return {"attributes": attr_logits, "malignancy": mal_logits}


def build_model(cfg) -> MultiHeadModel:
    return MultiHeadModel(cfg)

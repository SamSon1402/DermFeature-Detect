"""Post-hoc confidence calibration via temperature scaling (Guo et al., 2017).

A single scalar temperature ``T`` is fit on the validation set to minimise NLL;
dividing logits by ``T`` makes the predicted probabilities match observed
accuracy. We report ECE before and after so the improvement is measurable.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import expected_calibration_error


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        # parameterise as log T so T stays positive under unconstrained optim
        self.log_t = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> float:
        return float(self.log_t.exp().item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_t.exp()


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200):
    """Fit T on held-out logits/labels. Returns (scaler, ece_before, ece_after)."""
    logits = logits.detach().view(-1, 1).float()
    labels = labels.detach().view(-1, 1).float()
    scaler = TemperatureScaler()
    opt = torch.optim.LBFGS(scaler.parameters(), lr=0.05, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(scaler(logits), labels)
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        before = expected_calibration_error(torch.sigmoid(logits).numpy(), labels.numpy())
        after = expected_calibration_error(torch.sigmoid(scaler(logits)).numpy(), labels.numpy())
    return scaler, before, after

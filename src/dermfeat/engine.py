"""Training and evaluation loops (multi-task: malignancy + attributes)."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import metrics as M


def _to_device(target: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in target.items()}


def train_one_epoch(model, loader: DataLoader, optimizer, loss_fn, device,
                    scaler=None, grad_clip: float = 0.0, log_every: int = 20, epoch: int = 0):
    model.train()
    running = 0.0
    use_amp = scaler is not None
    for step, (images, target) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        target = _to_device(target, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(images)
            loss = loss_fn(out, target)
        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        running += loss.item()
        if log_every and step % log_every == 0:
            print(f"  epoch {epoch} step {step:4d}/{len(loader)}  loss {loss.item():.4f}")
    return running / max(1, len(loader))


@torch.no_grad()
def collect_logits(model, loader: DataLoader, device) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Return malignancy logits/labels and attribute prob/label arrays."""
    model.eval()
    mal_logits, mal_labels, attr_p, attr_y = [], [], [], []
    for images, target in loader:
        out = model(images.to(device))
        mal_logits.append(out["malignancy"].cpu())
        mal_labels.append(target["malignancy"])
        attr_p.append(torch.sigmoid(out["attributes"]).cpu().numpy())
        attr_y.append(target["attributes"].numpy())
    return (torch.cat(mal_logits), torch.cat(mal_labels),
            np.concatenate(attr_p), np.concatenate(attr_y))


@torch.no_grad()
def evaluate(model, loader: DataLoader, loss_fn, device, attributes) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    mal_logits, mal_labels, attr_p, attr_y = collect_logits(model, loader, device)
    for images, target in loader:
        out = model(images.to(device))
        total_loss += loss_fn(out, _to_device(target, device)).item()
    probs = torch.sigmoid(mal_logits).numpy()
    res = M.malignancy_report(probs, mal_labels.numpy())
    res["loss"] = total_loss / max(1, len(loader))
    # mean attribute AUROC (where a label varies); attributes are continuous in
    # the synthetic set, so threshold at 0.5 to score ranking quality
    aurocs = []
    for j in range(attr_y.shape[1]):
        aurocs.append(M.auroc(attr_p[:, j], (attr_y[:, j] > 0.5).astype(float)))
    res["attr_auroc"] = float(np.nanmean(aurocs)) if aurocs else float("nan")
    # selective risk at 70% coverage shows the abstention benefit
    res["risk@0.7cov"] = M.selective_risk(probs, mal_labels.numpy(), 0.7)["risk"]
    return res

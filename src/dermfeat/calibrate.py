"""Fit a temperature on the validation set and record it in the checkpoint.

    python -m dermfeat.calibrate --checkpoint runs/exp/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .calibration import fit_temperature
from .config import load_config
from .engine import collect_logits
from .models import build_model


def build_val_loader(cfg, device):
    if cfg.data.kind == "synthetic":
        from .data.synthetic import SyntheticLesionClsDataset

        n_val = max(1, int(cfg.data.synthetic_n * cfg.data.val_fraction))
        ds = SyntheticLesionClsDataset(n_val, cfg.data.image_size, cfg.model.attributes, seed=999)
    else:
        from .data.dataset import ISICClassificationDataset, split_indices
        from .data.transforms import build_transforms

        full = ISICClassificationDataset(cfg.data.root, cfg.data.metadata, cfg.model.attributes)
        _, va = split_indices(len(full), cfg.data.val_fraction, cfg.train.seed)
        ds = ISICClassificationDataset(cfg.data.root, cfg.data.metadata, cfg.model.attributes,
                                       build_transforms(cfg.data.image_size, False), va)
    return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = load_config(ckpt["config"])
    model = build_model(cfg.model).to(device)
    model.load_state_dict(ckpt["model"])

    loader = build_val_loader(cfg, device)
    logits, labels, _, _ = collect_logits(model, loader, device)
    scaler, ece_before, ece_after = fit_temperature(logits, labels)

    ckpt["temperature"] = scaler.temperature
    torch.save(ckpt, args.checkpoint)
    print(f"fitted temperature T = {scaler.temperature:.3f}")
    print(f"ECE before {ece_before:.4f}  ->  after {ece_after:.4f}")
    print(f"saved T into {args.checkpoint}")


if __name__ == "__main__":
    main()

"""Training entry point.

    python -m dermfeat.train --config configs/smoke.yaml     # seconds, CPU OK
    python -m dermfeat.train --config configs/isic.yaml      # full ISIC training
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config, save_config
from .engine import evaluate, train_one_epoch
from .losses import build_loss
from .models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_datasets(cfg):
    if cfg.data.kind == "synthetic":
        from .data.synthetic import SyntheticLesionClsDataset

        n = cfg.data.synthetic_n
        n_val = max(1, int(n * cfg.data.val_fraction))
        attrs = cfg.model.attributes
        train = SyntheticLesionClsDataset(n - n_val, cfg.data.image_size, attrs, seed=1)
        val = SyntheticLesionClsDataset(n_val, cfg.data.image_size, attrs, seed=999)
        return train, val

    if cfg.data.kind == "isic":
        from .data.dataset import ISICClassificationDataset, split_indices
        from .data.transforms import build_transforms

        full = ISICClassificationDataset(cfg.data.root, cfg.data.metadata, cfg.model.attributes)
        tr_idx, va_idx = split_indices(len(full), cfg.data.val_fraction, cfg.train.seed)
        train = ISICClassificationDataset(cfg.data.root, cfg.data.metadata, cfg.model.attributes,
                                          build_transforms(cfg.data.image_size, True), tr_idx)
        val = ISICClassificationDataset(cfg.data.root, cfg.data.metadata, cfg.model.attributes,
                                        build_transforms(cfg.data.image_size, False), va_idx)
        return train, val

    raise ValueError(f"Unknown data.kind '{cfg.data.kind}'.")


def make_scheduler(optimizer, cfg, steps_per_epoch):
    warmup = cfg.train.warmup_epochs * steps_per_epoch
    total = cfg.train.epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    ap = argparse.ArgumentParser(description="Train the multi-head lesion classifier.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    device = torch.device(args.device)
    out = Path(cfg.train.out_dir); out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")

    train_set, val_set = build_datasets(cfg)
    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=cfg.train.num_workers, pin_memory=device.type == "cuda",
                              drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg.train.batch_size, shuffle=False,
                            num_workers=cfg.train.num_workers, pin_memory=device.type == "cuda")

    model = build_model(cfg.model).to(device)
    loss_fn = build_loss(cfg.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    steps = max(1, len(train_set) // cfg.train.batch_size)
    scheduler = make_scheduler(optimizer, cfg, steps)
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"backbone={cfg.model.backbone}  mode={cfg.model.malignancy_mode}  params={n_params:.1f}M  "
          f"device={device}  train={len(train_set)}  val={len(val_set)}")

    csv_path = out / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "auroc", "accuracy",
                                "sensitivity", "specificity", "ece", "attr_auroc", "risk@0.7cov", "lr"])

    best, best_epoch, since = -1.0, -1, 0
    metric = cfg.train.select_metric
    for epoch in range(cfg.train.epochs):
        tr = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler,
                             cfg.train.grad_clip, cfg.train.log_every, epoch)
        for _ in range(len(train_loader)):
            scheduler.step()
        val = evaluate(model, val_loader, loss_fn, device, cfg.model.attributes)
        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:3d}  train {tr:.4f}  val {val['loss']:.4f}  "
              f"auroc {val['auroc']:.4f}  acc {val['accuracy']:.4f}  "
              f"sens {val['sensitivity']:.3f}  spec {val['specificity']:.3f}  ece {val['ece']:.4f}")
        with csv_path.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tr:.5f}", f"{val['loss']:.5f}", f"{val['auroc']:.5f}",
                                    f"{val['accuracy']:.5f}", f"{val['sensitivity']:.5f}",
                                    f"{val['specificity']:.5f}", f"{val['ece']:.5f}",
                                    f"{val['attr_auroc']:.5f}", f"{val['risk@0.7cov']:.5f}", f"{lr:.2e}"])

        score = val.get(metric, val["auroc"])
        if score is not None and score == score and score > best:
            best, best_epoch, since = score, epoch, 0
            torch.save({"model": model.state_dict(), "config": args.config,
                        "epoch": epoch, metric: best}, out / "best.pt")
        else:
            since += 1
            if since >= cfg.train.early_stop_patience:
                print(f"early stop after {since} epochs without improvement")
                break

    (out / "summary.json").write_text(json.dumps({f"best_{metric}": best, "best_epoch": best_epoch}, indent=2))
    print(f"done. best {metric} {best:.4f} @ epoch {best_epoch}. "
          f"checkpoint -> {out/'best.pt'}. Next: python -m dermfeat.calibrate --checkpoint {out/'best.pt'}")


if __name__ == "__main__":
    main()

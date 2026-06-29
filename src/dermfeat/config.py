"""Typed configuration (dataclass tree loaded from YAML)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

DEFAULT_ATTRIBUTES = ["asymmetry", "border", "color", "diameter"]


@dataclass
class DataConfig:
    kind: str = "synthetic"           # "isic" | "synthetic"
    root: Optional[str] = None
    metadata: Optional[str] = None     # CSV/JSON with labels for kind="isic"
    image_size: int = 224
    val_fraction: float = 0.2
    synthetic_n: int = 400


@dataclass
class ModelConfig:
    backbone: str = "resnet_scratch"   # "resnet_scratch" | any timm name
    in_channels: int = 3
    width: int = 32                    # channel width for the scratch ResNet
    pretrained: bool = True            # used by timm backbones
    dropout: float = 0.2
    attributes: List[str] = field(default_factory=lambda: list(DEFAULT_ATTRIBUTES))
    malignancy_mode: str = "fused"     # "fused" | "direct"


@dataclass
class LossConfig:
    w_malignancy: float = 1.0
    w_attributes: float = 0.5
    focal_gamma: float = 2.0
    pos_weight: Optional[float] = 3.0  # up-weight the rare malignant class


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    amp: bool = True
    grad_clip: float = 1.0
    warmup_epochs: int = 2
    early_stop_patience: int = 10
    seed: int = 1402
    out_dir: str = "runs/exp"
    log_every: int = 20
    select_metric: str = "auroc"       # checkpoint on this val metric


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _merge(dc, overrides: dict):
    for key, val in (overrides or {}).items():
        if not hasattr(dc, key):
            raise KeyError(f"Unknown config key: {type(dc).__name__}.{key}")
        cur = getattr(dc, key)
        if dataclasses.is_dataclass(cur) and isinstance(val, dict):
            _merge(cur, val)
        else:
            setattr(dc, key, val)
    return dc


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return _merge(Config(), raw)


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False))

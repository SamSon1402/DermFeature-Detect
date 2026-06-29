"""A compact ResNet backbone implemented from scratch.

Written out rather than imported so the residual structure is visible. Returns a
pooled feature vector. ``build_backbone`` can instead use a pretrained ``timm``
encoder when that package is installed (the production choice).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNetBackbone(nn.Module):
    """ResNet-18-style backbone. ``feature_dim`` is the pooled output width."""

    def __init__(self, in_ch: int = 3, width: int = 64, blocks=(2, 2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, 7, 2, 3, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        chans = [width, width * 2, width * 4, width * 8]
        self.layer1 = self._stage(width, chans[0], blocks[0], 1)
        self.layer2 = self._stage(chans[0], chans[1], blocks[1], 2)
        self.layer3 = self._stage(chans[1], chans[2], blocks[2], 2)
        self.layer4 = self._stage(chans[2], chans[3], blocks[3], 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = chans[3]

    @staticmethod
    def _stage(in_ch, out_ch, n, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        layers += [BasicBlock(out_ch, out_ch, 1) for _ in range(n - 1)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return torch.flatten(self.pool(x), 1)


class TimmBackbone(nn.Module):  # pragma: no cover - optional path
    def __init__(self, name: str, in_ch: int = 3, pretrained: bool = True):
        super().__init__()
        import timm

        self.net = timm.create_model(name, pretrained=pretrained, in_chans=in_ch,
                                     num_classes=0, global_pool="avg")
        self.feature_dim = self.net.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_backbone(cfg) -> nn.Module:
    if cfg.backbone == "resnet_scratch":
        return ResNetBackbone(in_ch=cfg.in_channels, width=cfg.width)
    return TimmBackbone(cfg.backbone, in_ch=cfg.in_channels, pretrained=cfg.pretrained)

import torch

from dermfeat.calibration import fit_temperature
from dermfeat.config import ModelConfig, LossConfig
from dermfeat.losses import build_loss
from dermfeat.models import build_model


def _cfg(**kw):
    base = dict(backbone="resnet_scratch", width=8, dropout=0.0,
                attributes=["asymmetry", "border", "color", "diameter"],
                malignancy_mode="fused")
    base.update(kw)
    return ModelConfig(**base)


def test_model_output_shapes():
    for mode in ("fused", "direct"):
        model = build_model(_cfg(malignancy_mode=mode))
        out = model(torch.randn(3, 3, 64, 64))
        assert out["attributes"].shape == (3, 4)
        assert out["malignancy"].shape == (3, 1)


def test_multitask_loss_scalar():
    model = build_model(_cfg())
    out = model(torch.randn(2, 3, 64, 64))
    target = {"malignancy": torch.tensor([[1.0], [0.0]]),
              "attributes": torch.rand(2, 4)}
    loss = build_loss(LossConfig())(out, target)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_temperature_scaling_improves_ece():
    torch.manual_seed(0)
    # over-confident logits (too large) -> temperature should soften them
    logits = torch.randn(500, 1) * 4.0
    labels = (torch.sigmoid(logits / 3.0) > 0.5).float()
    scaler, before, after = fit_temperature(logits, labels)
    assert after <= before + 1e-6
    assert scaler.temperature > 0

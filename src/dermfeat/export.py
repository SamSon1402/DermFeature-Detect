"""Export a trained checkpoint to TorchScript and ONNX.

    python -m dermfeat.export --checkpoint runs/exp/best.pt --out export/

The model returns a dict; for export we wrap it so the graph emits two named
tensors (attribute logits, malignancy logits).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from .config import load_config
from .models import build_model


class _ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["attributes"], out["malignancy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="export")
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = load_config(ckpt["config"])
    size = args.image_size or cfg.data.image_size
    model = build_model(cfg.model)
    model.load_state_dict(ckpt["model"])
    model.eval()
    wrapped = _ExportWrapper(model).eval()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, cfg.model.in_channels, size, size)

    ts_path = out / "model.ts.pt"
    torch.jit.trace(wrapped, dummy).save(str(ts_path))
    print(f"TorchScript -> {ts_path}")

    onnx_path = out / "model.onnx"
    kwargs = dict(input_names=["image"], output_names=["attributes", "malignancy"],
                  dynamic_axes={"image": {0: "batch"}, "attributes": {0: "batch"},
                                "malignancy": {0: "batch"}},
                  opset_version=args.opset)
    try:
        torch.onnx.export(wrapped, dummy, str(onnx_path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(wrapped, dummy, str(onnx_path), **kwargs)
    print(f"ONNX        -> {onnx_path}")

    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            ref_a, ref_m = wrapped(dummy)
        got_a, got_m = sess.run(None, {"image": dummy.numpy()})
        err = max(float(np.abs(ref_a.numpy() - got_a).max()),
                  float(np.abs(ref_m.numpy() - got_m).max()))
        print(f"ONNX parity check: max abs error {err:.2e} ({'OK' if err < 1e-3 else 'CHECK'})")
    except ImportError:
        print("onnxruntime not installed - skipped parity check")


if __name__ == "__main__":
    main()

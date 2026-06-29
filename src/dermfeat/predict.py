"""Run a trained checkpoint on one image: attribute probabilities, a calibrated
malignancy probability, and a triage decision (with abstention).

    python -m dermfeat.predict --checkpoint runs/exp/best.pt --image lesion.jpg
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .abstention import triage
from .config import load_config
from .models import build_model


@torch.no_grad()
def infer(model, image: np.ndarray, image_size: int, device, temperature: float = 1.0):
    import cv2

    inp = cv2.resize(image, (image_size, image_size)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    inp = (inp - mean) / std
    tensor = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).float().to(device)
    out = model(tensor)
    attr = torch.sigmoid(out["attributes"])[0].cpu().numpy()
    mal_prob = torch.sigmoid(out["malignancy"] / temperature)[0, 0].item()
    return attr, mal_prob


def main():
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = load_config(ckpt["config"])
    model = build_model(cfg.model).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    temperature = ckpt.get("temperature", 1.0)

    image = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    attr, mal_prob = infer(model, image, cfg.data.image_size, device, temperature)
    t = triage(mal_prob)

    result = {
        "attributes": {name: round(float(p), 3) for name, p in zip(cfg.model.attributes, attr)},
        "malignancy_probability": round(mal_prob, 3),
        "temperature": round(float(temperature), 3),
        "predictive_entropy": round(t.entropy, 3),
        "decision": t.decision,
        "abstain": t.abstain,
        "recommendation": t.recommendation,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

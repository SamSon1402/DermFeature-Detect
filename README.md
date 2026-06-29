# DermFeature-Detect

**Multi-head dermoscopy classifier in PyTorch — per-attribute (ABCD) heads feeding a calibrated malignancy head, with abstention.**

A malignancy probability on its own isn't enough to put in front of a clinician. This model is built the way a safety-critical classifier should be:

1. **Clinical attribute heads** — a shared backbone predicts the image-derivable ABCD signs (Asymmetry, Border, Colour, Diameter), and the malignancy decision is a learned, interpretable function of those signs.
2. **Calibration** — temperature scaling makes the reported confidence match observed accuracy (low ECE), so a "78%" means something.
3. **Abstention** — when predictive uncertainty is high the system defers to a clinician instead of guessing.

The backbone is a ResNet implemented from scratch so the architecture is auditable; an optional `timm` backbone is available for the production path. The whole pipeline — including calibration — runs end-to-end on synthetic data in seconds before you point it at real ISIC images.

> On the fifth classic sign, *Evolving*: it's temporal and can't come from a single image, so it's handled by the longitudinal tracker (project 3), not faked as a single-image head here.

---

## Why these choices

| Decision | Reason |
|---|---|
| **Multi-task (attributes + malignancy)** | Supervising the clinical signs gives the malignancy head better, more interpretable features and a built-in explanation of *why*. |
| **Fused malignancy head** | `malignancy = linear(attribute_logits)` — the risk is an explicit function of the signs, mirroring how a clinician reasons (toggle to a `direct` head in config). |
| **Focal loss + class weight** | Malignant cases are rare; focal loss stops the benign majority from dominating the gradient. |
| **Temperature scaling** | Cheap, effective post-hoc calibration (Guo et al. 2017); we report ECE before/after so the gain is measurable. |
| **Selective prediction** | Reports risk at a coverage level and abstains under high entropy — the honest behaviour for a clinical tool. |

## Metrics

AUROC, accuracy, **sensitivity / specificity** (the numbers a clinician cares about), **ECE** (calibration), mean attribute AUROC, and **selective risk @ coverage** (the abstention benefit).

---

## Quickstart

```bash
pip install -e .                                  # core deps

python -m dermfeat.train     --config configs/smoke.yaml      # ~seconds, CPU ok
python -m dermfeat.calibrate --checkpoint runs/smoke/best.pt  # fit temperature, report ECE
pytest -q
```

A successful smoke run trains the multi-head model on synthetic data (AUROC
rises on a held-out split), then calibration reports ECE before vs after.

## Training on real ISIC data

```bash
pip install -e ".[isic]"
python scripts/download_isic.py --out data/isic --limit 2000   # images + metadata.json (benign/malignant)
python -m dermfeat.train --config configs/isic.yaml
python -m dermfeat.calibrate --checkpoint runs/isic/best.pt
```

The ISIC API provides the `benign_malignant` label used for the malignancy head.
Attribute supervision is optional — if your metadata has attribute columns
(e.g. from derm7pt) they're picked up automatically; otherwise only the
malignancy head is supervised.

Switch `model.backbone` to a `timm` name (e.g. `efficientnet_b0`, with
`pip install timm`) for a pretrained encoder.

## Inference

```bash
python -m dermfeat.predict --checkpoint runs/isic/best.pt --image lesion.jpg
```

```json
{
  "attributes": {"asymmetry": 0.81, "border": 0.74, "color": 0.69, "diameter": 0.62},
  "malignancy_probability": 0.78,
  "predictive_entropy": 0.76,
  "decision": "refer",
  "abstain": false,
  "recommendation": "Refer to a dermatologist; biopsy decision is the clinician's."
}
```

## Export

```bash
python -m dermfeat.export --checkpoint runs/isic/best.pt --out export/
```

Writes `model.onnx` (dynamic batch, two named outputs) and `model.ts.pt`, with an
ONNX↔PyTorch parity check when `onnxruntime` is installed.

---

## Layout

```
src/dermfeat/
  config.py            typed dataclass config (YAML)
  metrics.py           AUROC, sens/spec, ECE, reliability, selective risk
  losses.py            focal loss + multi-task loss
  calibration.py       temperature scaling (fit on val)
  abstention.py        triage decision + abstain-on-uncertainty
  engine.py            train / evaluate (multi-task)
  train.py             training CLI
  calibrate.py         fit + store temperature
  predict.py           image -> attributes + calibrated risk + triage
  export.py            ONNX + TorchScript
  models/
    backbone.py        from-scratch ResNet (+ optional timm)
    heads.py           multi-head model (fused / direct malignancy)
  data/
    dataset.py         ISIC classification dataset
    transforms.py      albumentations
    synthetic.py       on-the-fly data for the smoke run
scripts/download_isic.py
tests/                 metrics, model shapes, calibration
configs/{smoke,isic}.yaml
```

## Data & license

Dermoscopy images and labels from the [ISIC Archive](https://www.isic-archive.com)
(CC-licensed). Code is MIT (see `LICENSE`).

> The synthetic generator is for pipeline testing only; reported clinical-grade
> numbers should come from training on real ISIC labels.

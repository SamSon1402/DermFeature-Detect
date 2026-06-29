.PHONY: setup smoke test train calibrate export
PY ?= python

setup:
	pip install -e .

smoke:      ## end-to-end check on synthetic data (seconds, CPU ok)
	PYTHONPATH=src $(PY) -m dermfeat.train --config configs/smoke.yaml

test:
	PYTHONPATH=src pytest -q

train:      ## full ISIC training (needs data/isic + extras)
	PYTHONPATH=src $(PY) -m dermfeat.train --config configs/isic.yaml

calibrate:  ## fit temperature on the validation set
	PYTHONPATH=src $(PY) -m dermfeat.calibrate --checkpoint runs/isic/best.pt

export:     ## export best checkpoint to ONNX + TorchScript
	PYTHONPATH=src $(PY) -m dermfeat.export --checkpoint runs/isic/best.pt --out export

import numpy as np

from dermfeat.metrics import (auroc, expected_calibration_error,
                              selective_risk, sensitivity_specificity)


def test_auroc_perfect_and_random():
    labels = np.array([0, 0, 1, 1])
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 1.0
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == 0.0
    # constant scores -> chance
    assert abs(auroc(np.array([0.5, 0.5, 0.5, 0.5]), labels) - 0.5) < 1e-9


def test_sensitivity_specificity():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    sens, spec = sensitivity_specificity(scores, labels)
    assert sens == 1.0 and spec == 1.0


def test_ece_zero_for_perfectly_calibrated():
    # confident & correct everywhere -> ECE ~ 0
    p = np.array([0.99, 0.98, 0.01, 0.02])
    y = np.array([1, 1, 0, 0])
    assert expected_calibration_error(p, y) < 0.05


def test_selective_risk_drops_with_lower_coverage():
    # most-confident predictions are correct; least-confident are wrong
    p = np.array([0.99, 0.95, 0.55, 0.52])
    y = np.array([1, 1, 0, 1])   # the 0.52 one (pred 1) is correct? -> craft errors
    full = selective_risk(p, y, 1.0)["risk"]
    top = selective_risk(p, y, 0.5)["risk"]
    assert top <= full

import numpy as np

from stroke_ai.calibration import Calibrator
from stroke_ai.metrics import (bootstrap_ci, compute_metrics, expected_calibration_error,
                               paired_bootstrap_auc_diff, sensitivity_at_specificity, youden_threshold)


def test_perfect_classifier():
    y = np.array([0, 0, 0, 1, 1]); p = np.array([.1, .2, .3, .8, .9])
    m = compute_metrics(y, p, 0.5)
    assert m["roc_auc"] == 1.0 and m["sensitivity"] == 1.0 and m["specificity"] == 1.0 and m["mcc"] == 1.0
    assert sensitivity_at_specificity(y, p, 0.9) == 1.0


def test_accuracy_is_misleading_on_imbalance():
    y = np.array([0] * 95 + [1] * 5); p = np.zeros(100)
    m = compute_metrics(y, p, 0.5)
    assert m["accuracy"] == 0.95 and m["sensitivity"] == 0.0 and m["mcc"] == 0.0


def test_known_confusion_counts():
    y = np.array([1, 1, 0, 0]); p = np.array([.9, .2, .8, .1])
    m = compute_metrics(y, p, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)


def test_youden_picks_separating_threshold():
    y = np.array([0, 0, 0, 1, 1, 1]); p = np.array([.1, .2, .3, .6, .7, .8])
    t = youden_threshold(y, p)
    assert 0.3 < t <= 0.6


def test_ece_zero_for_perfectly_calibrated_extremes():
    assert expected_calibration_error(np.array([0, 1]), np.array([0.0, 1.0])) == 0.0


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300); p = np.clip(y * 0.3 + rng.normal(0.4, 0.2, 300), 0, 1)
    lo, hi = bootstrap_ci(y, p, "roc_auc", 200, 0)
    auc = compute_metrics(y, p)["roc_auc"]
    assert lo <= auc <= hi
    d = paired_bootstrap_auc_diff(y, p, p, 100, 0)
    assert d["auc_diff"] == 0 and d["ci_low"] == 0 and d["ci_high"] == 0


def test_platt_calibration_reduces_ece_on_overconfident_scores():
    rng = np.random.default_rng(1)
    true_p = rng.uniform(0.05, 0.6, 4000); y = (rng.random(4000) < true_p).astype(int)
    overconf = np.clip(true_p * 1.0 + (true_p - 0.3) * 1.5, 0.01, 0.99)       # distorted but monotone
    cal = Calibrator("sigmoid").fit(y[:2000], overconf[:2000])
    assert expected_calibration_error(y[2000:], cal.transform(overconf[2000:])) < expected_calibration_error(y[2000:], overconf[2000:])

"""Medical-ML evaluation metrics. Accuracy is reported but never the headline."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)

_BOOT_METRICS = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score}


def youden_threshold(y, p) -> float:
    """Threshold maximising sensitivity + specificity - 1. Choose it on VALIDATION data only."""
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    t = thr[int(np.argmax(j))]
    return float(min(t, 1.0)) if np.isfinite(t) else 1.0


def sensitivity_at_specificity(y, p, target_spec: float = 0.90) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    ok = (1 - fpr) >= target_spec
    return float(tpr[ok].max()) if ok.any() else float("nan")


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def compute_metrics(y, p, threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if both else float("nan"),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(f1),
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 or len(np.unique(y)) > 1 else 0.0,
        "accuracy": float((tp + tn) / len(y)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "sens_at_90_spec": sensitivity_at_specificity(y, p, 0.90) if both else float("nan"),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def _stratified_resample(y, rng):
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    return np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def bootstrap_ci(y, p, metric: str = "roc_auc", n_boot: int = 500, seed: int = 0, alpha: float = 0.05):
    """Stratified percentile bootstrap CI for 'roc_auc' or 'pr_auc'. With few positives the CI is wide: that is honest."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    fn = _BOOT_METRICS[metric]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = _stratified_resample(y, rng)
        vals.append(fn(y[i], p[i]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap_auc_diff(y, p_a, p_b, n_boot: int = 500, seed: int = 0, alpha: float = 0.05) -> dict:
    """AUC(a) - AUC(b) on the same patients, with a percentile CI. If the CI excludes 0 the gap is unlikely to be resampling noise."""
    y = np.asarray(y).astype(int)
    p_a, p_b = np.asarray(p_a, float), np.asarray(p_b, float)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = _stratified_resample(y, rng)
        diffs.append(roc_auc_score(y[i], p_a[i]) - roc_auc_score(y[i], p_b[i]))
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "auc_diff": float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b)),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }

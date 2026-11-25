"""Robustness tests: missing clinical values and degraded images. Perturb TEST data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

from .data.clinical import CLINICAL_COLUMNS, ensure_clinical_columns


def mask_missing(df: pd.DataFrame, frac: float, seed: int = 0, columns=None) -> pd.DataFrame:
    """Set each clinical cell to NaN independently with probability `frac`."""
    df = ensure_clinical_columns(df)
    cols = list(columns or CLINICAL_COLUMNS)
    rng = np.random.default_rng(seed)
    hide = rng.random((len(df), len(cols))) < frac
    out = df.copy()
    out[cols] = df[cols].mask(hide)
    return out


def degrade_images(imgs: np.ndarray, kind: str, level: float, seed: int = 0) -> np.ndarray:
    """kind: noise (sigma) | blur (sigma, px) | contrast (factor<1 lowers) | lowres (integer factor)."""
    x = imgs.astype(np.float32).copy()
    if kind == "noise":
        x = x + np.random.default_rng(seed).normal(0, level, x.shape).astype(np.float32)
    elif kind == "blur":
        if level > 0:
            x = gaussian_filter(x, sigma=(0, 0, level, level))
    elif kind == "contrast":
        m = x.mean(axis=(2, 3), keepdims=True)
        x = (x - m) * level + m
    elif kind == "lowres":
        k = int(level)
        if k > 1:
            h, w = x.shape[-2:]
            small = x[:, :, ::k, ::k]
            x = np.repeat(np.repeat(small, k, axis=2), k, axis=3)[:, :, :h, :w]
    else:
        raise ValueError(f"unknown degradation {kind!r}")
    return np.clip(x, 0.0, 1.0).astype(np.float32)


IMAGE_SWEEPS = {
    "noise": [0.0, 0.05, 0.1, 0.2],
    "blur": [0.0, 1.0, 2.0, 3.0],
    "contrast": [1.0, 0.7, 0.5, 0.3],
    "lowres": [1, 2, 4],
}
MISSING_LEVELS = [0.0, 0.1, 0.2, 0.3]


def robustness_table(scorers: dict, df: pd.DataFrame, imgs: np.ndarray, y, seed: int = 0) -> pd.DataFrame:
    """AUC of each named Scorer under each perturbation level. Returns a long DataFrame."""
    y = np.asarray(y).astype(int)
    rows = []
    for name, sc in scorers.items():
        for frac in MISSING_LEVELS:
            p = sc.predict(mask_missing(df, frac, seed), imgs)
            rows.append({"model": name, "perturbation": "missing_clinical", "level": frac, "roc_auc": roc_auc_score(y, p)})
        if sc.kind == "clinical":
            continue  # image perturbations cannot change a clinical-only model
        for kind, levels in IMAGE_SWEEPS.items():
            for lv in levels:
                p = sc.predict(df, degrade_images(imgs, kind, lv, seed))
                rows.append({"model": name, "perturbation": kind, "level": lv, "roc_auc": roc_auc_score(y, p)})
    return pd.DataFrame(rows)

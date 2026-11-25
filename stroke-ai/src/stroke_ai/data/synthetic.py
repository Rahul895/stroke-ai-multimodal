"""Synthetic paired cohort (clinical table + MRI-like slices).

PURPOSE: smoke-testing and demonstrating the pipeline end to end. The images are
NOT real MRI and the clinical risk model is invented. Results on this data say
nothing about real-world stroke prediction.

Each patient has BOTH modalities by construction. The label depends on the
clinical variables (noisy), and the image shows a lesion-like blob for most
positives and a few negatives, so the two modalities carry partly independent
information about the label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from ..utils import sigmoid

SMOKING = ["never smoked", "formerly smoked", "smokes", "Unknown"]


def _background(n: int, size: int, rng, brightness: float, noise: float):
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2
    r = np.sqrt(((yy - c) / (0.42 * size)) ** 2 + ((xx - c) / (0.36 * size)) ** 2)
    brain = (r < 1.0).astype(np.float32)
    skull = ((r >= 1.0) & (r < 1.12)).astype(np.float32)
    base = 0.45 * brain + 0.8 * skull
    imgs = np.empty((n, 1, size, size), dtype=np.float32)
    for i in range(n):
        texture = gaussian_filter(rng.normal(0, 1, (size, size)), sigma=3) * 0.08
        imgs[i, 0] = base * (1 + texture) + brightness + rng.normal(0, noise, (size, size))
    return imgs, brain


def _add_lesion(img: np.ndarray, brain: np.ndarray, rng, size: int) -> None:
    ys, xs = np.nonzero(brain > 0)
    k = rng.integers(len(ys))
    cy, cx = ys[k], xs[k]
    sigma = rng.uniform(0.04, 0.09) * size
    amp = rng.uniform(0.2, 0.45)
    yy, xx = np.mgrid[0:size, 0:size]
    img[0] += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)) * brain


def make_synthetic_cohort(
    n: int = 800,
    img_size: int = 64,
    seed: int = 42,
    shift: bool = False,
    lesion_visibility: float = 0.8,
    artifact_rate: float = 0.06,
):
    """Return (clinical_df, images). images: float32 (n, 1, H, W) in [0, 1].

    shift=True mimics a different site: older patients, higher glucose, brighter and
    noisier scans. Used to demonstrate external validation code.
    """
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(62 if shift else 55, 16, n), 18, 90).round()
    gender = rng.choice(["Male", "Female"], n)
    hyp = (rng.random(n) < np.where(age > 60, 0.5, 0.1)).astype(int)
    hd = (rng.random(n) < np.where(age > 60, 0.19, 0.04)).astype(int)
    glucose = np.clip(rng.normal(110 if shift else 100, 25, n) + 20 * hyp, 55, 300).round(1)
    bmi = np.clip(rng.normal(28, 6, n), 15, 55).round(1)
    bmi[rng.random(n) < 0.04] = np.nan
    smoking = rng.choice(SMOKING, n, p=[0.5, 0.2, 0.15, 0.15])

    bmi_filled = np.nan_to_num(bmi, nan=28.0)
    z = (
        -2.0
        + 0.045 * (age - 55)
        + 0.9 * hyp
        + 0.7 * hd
        + 0.010 * (glucose - 100)
        + 0.5 * (smoking == "smokes")
        + 0.15 * (bmi_filled - 28) / 6
    )
    stroke = (rng.random(n) < sigmoid(z)).astype(int)

    imgs, brain = _background(
        n, img_size, rng, brightness=0.04 if shift else 0.0, noise=0.08 if shift else 0.05
    )
    for i in range(n):
        p_lesion = lesion_visibility if stroke[i] else artifact_rate
        if rng.random() < p_lesion:
            _add_lesion(imgs[i], brain, rng, img_size)
    imgs = np.clip(imgs, 0.0, 1.0).astype(np.float32)

    prefix = "X" if shift else "P"
    df = pd.DataFrame(
        {
            "patient_id": [f"{prefix}{i:05d}" for i in range(n)],
            "gender": gender,
            "age": age,
            "hypertension": hyp,
            "heart_disease": hd,
            "avg_glucose_level": glucose,
            "bmi": bmi,
            "smoking_status": smoking,
            "stroke": stroke,
        }
    )
    return df, imgs

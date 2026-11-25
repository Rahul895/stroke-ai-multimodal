"""Image loading and normalisation. Images are kept as float32 in [0, 1], shape (1, H, W)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .nifti import representative_axial_slice, load_nifti


def load_image(src, size: int = 64) -> np.ndarray:
    """Load PNG/JPG, .npy, or NIfTI (.nii/.nii.gz) as grayscale (1, size, size).

    NIfTI support is label-free and selects one representative axial slice. For
    research experiments with a particular DWI/ADC protocol, cache a dataset-
    specific slice/2.5-D representation instead of relying on this default.
    """
    if isinstance(src, (str, Path)):
        low = str(src).lower()
        if low.endswith((".nii", ".nii.gz")):
            arr = representative_axial_slice(load_nifti(src)).astype(np.float32)
            lo, hi = float(np.percentile(arr, 1)), float(np.percentile(arr, 99))
            arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0.0, 1.0)
            im = Image.fromarray(arr, mode="F").resize((size, size), Image.BILINEAR)
            return np.asarray(im, dtype=np.float32)[None]
        if low.endswith(".npy"):
            arr = np.squeeze(np.load(src).astype(np.float32))
            if arr.ndim != 2:
                raise ValueError(f"{src}: expected a 2-D slice, got shape {arr.shape}. Extract a slice first.")
            lo, hi = float(arr.min()), float(arr.max())
            arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
            im = Image.fromarray(arr, mode="F").resize((size, size), Image.BILINEAR)
            return np.asarray(im, dtype=np.float32)[None]
    im = Image.open(src).convert("L").resize((size, size), Image.BILINEAR)
    return (np.asarray(im, dtype=np.float32) / 255.0)[None]


def normalize(imgs: np.ndarray) -> np.ndarray:
    """Per-scan standardisation (zero mean, unit variance within each image).

    Done per image, not with dataset-wide statistics, so a global brightness/gain offset between
    scanners or sites does not shift the network's inputs. Nothing is learned from the data here,
    so there is no train/test leakage and nothing to store alongside the model.
    """
    imgs = np.asarray(imgs, dtype=np.float32)
    m = imgs.mean(axis=(1, 2, 3), keepdims=True)
    s = imgs.std(axis=(1, 2, 3), keepdims=True) + 1e-6
    return ((imgs - m) / s).astype(np.float32)

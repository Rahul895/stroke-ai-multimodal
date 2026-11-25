"""NIfTI utilities for real MRI datasets.

The research pipeline intentionally keeps its model input as a 2-D slice. This module
converts a 3-D/4-D NIfTI volume into a reproducible representative axial slice and
can cache the result as .npy for fast repeated experiments.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _require_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - depends on optional real-data install
        raise ImportError(
            "Real NIfTI support requires nibabel. Install with: pip install -r requirements-real.txt"
        ) from exc
    return nib


def load_nifti(path: str | Path) -> np.ndarray:
    nib = _require_nibabel()
    arr = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32))
    arr = np.squeeze(arr)
    if arr.ndim == 4:
        # For a multi-volume scan, use the first volume by default. The dataset-specific
        # preprocessing script can be adapted if a particular DWI/ADC volume is required.
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected a 3-D MRI volume after squeeze, got {arr.shape}")
    return arr


def representative_axial_slice(volume: np.ndarray) -> np.ndarray:
    """Choose an informative axial slice without using labels.

    Empty/background-heavy slices are ignored. The selected slice maximises the
    number of non-background voxels after a robust percentile threshold.
    """
    v = np.nan_to_num(np.asarray(volume, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    # Robust foreground threshold. This is deliberately label-free.
    nz = np.abs(v) > np.percentile(np.abs(v), 35)
    scores = nz.reshape(v.shape[0] * v.shape[1], v.shape[2]).sum(axis=0)
    # Avoid selecting extreme edge slices when several are similar.
    valid = np.arange(v.shape[2])
    if len(valid) >= 8:
        valid = valid[2:-2]
    idx = int(valid[np.argmax(scores[valid])])
    return v[:, :, idx]


def nifti_to_npy(src: str | Path, dst: str | Path) -> Path:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, representative_axial_slice(load_nifti(src)).astype(np.float32))
    return dst

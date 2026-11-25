"""Loader for a REAL paired dataset described by one CSV.

Required columns: patient_id (or id), stroke (0/1), image_path, plus the clinical columns in
data/clinical.py. One row = one (clinical record, scan) pair. A patient may have several rows;
splits are always done per patient.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .clinical import coerce_clinical_frame
from .imaging import load_image


def load_paired_csv(csv_path: str, image_root: str | None = None, img_size: int = 64):
    df = pd.read_csv(csv_path)
    if "image_path" not in df.columns:
        raise ValueError("CSV needs an 'image_path' column.")
    df = coerce_clinical_frame(df)
    root = Path(image_root) if image_root else Path(csv_path).parent
    imgs = np.stack([load_image(root / p, img_size) for p in df["image_path"]]).astype(np.float32)
    return df.reset_index(drop=True), imgs

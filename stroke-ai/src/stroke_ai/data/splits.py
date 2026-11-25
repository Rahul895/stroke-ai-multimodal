"""Patient-level splitting. No patient ever appears in more than one split."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def patient_level_split(
    df: pd.DataFrame,
    label_col: str = "stroke",
    patient_col: str = "patient_id",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Stratified split at PATIENT level. Returns positional row indices per split.

    A patient with several scans/rows keeps all of them in one split. A patient counts as
    positive if any of their rows is positive (used for stratification only).
    """
    df = df.reset_index(drop=True)
    pat = df.groupby(patient_col)[label_col].max()
    ids = pat.index.to_numpy()
    lab = pat.to_numpy().astype(int)
    if min(np.bincount(lab, minlength=2)) < 6:
        raise ValueError("Too few patients in one class to make stratified train/val/test splits.")

    trval_ids, test_ids, trval_lab, _ = train_test_split(
        ids, lab, test_size=test_frac, stratify=lab, random_state=seed
    )
    rel_val = val_frac / (1.0 - test_frac)
    train_ids, val_ids = train_test_split(
        trval_ids, test_size=rel_val, stratify=trval_lab, random_state=seed
    )
    sets = {"train": set(train_ids), "val": set(val_ids), "test": set(test_ids)}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
    return {k: np.flatnonzero(df[patient_col].isin(v).to_numpy()) for k, v in sets.items()}

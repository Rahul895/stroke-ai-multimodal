"""Subgroup performance. This reports whether performance VARIES across groups; it does not certify fairness."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import compute_metrics


def default_groups(df: pd.DataFrame) -> dict[str, np.ndarray]:
    g = {
        "age < 50": (df["age"] < 50).to_numpy(),
        "age >= 50": (df["age"] >= 50).to_numpy(),
        "female": (df["gender"] == "Female").to_numpy(),
        "male": (df["gender"] == "Male").to_numpy(),
        "hypertension": (df["hypertension"] == 1).to_numpy(),
        "no hypertension": (df["hypertension"] == 0).to_numpy(),
    }
    return g


def subgroup_table(df: pd.DataFrame, y, p, threshold: float, groups: dict | None = None, min_pos: int = 5) -> pd.DataFrame:
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    rows = []
    for name, mask in (groups or default_groups(df)).items():
        mask = np.asarray(mask, bool)
        n, npos = int(mask.sum()), int(y[mask].sum())
        row = {"group": name, "n": n, "n_pos": npos}
        if n == 0:
            rows.append(row)
            continue
        m = compute_metrics(y[mask], p[mask], threshold)
        row.update(sensitivity=m["sensitivity"], specificity=m["specificity"], roc_auc=m["roc_auc"])
        row["note"] = f"fewer than {min_pos} positives: unreliable" if npos < min_pos else ""
        rows.append(row)
    return pd.DataFrame(rows)

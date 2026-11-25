"""Clinical (tabular) schema and leakage-safe preprocessing.

Column names follow the common Kaggle "Healthcare Stroke Data" layout
(gender, age, hypertension, heart_disease, avg_glucose_level, bmi, smoking_status, stroke).
If your data uses other columns, change the three lists below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["age", "avg_glucose_level", "bmi"]
BINARY = ["hypertension", "heart_disease"]
CATEGORICAL = ["gender", "smoking_status"]
CLINICAL_COLUMNS = NUMERIC + BINARY + CATEGORICAL
LABEL = "stroke"
PATIENT_ID = "patient_id"


def build_preprocessor() -> ColumnTransformer:
    """Unfitted preprocessor. It must only ever be fitted on TRAINING rows
    (it lives inside the model pipelines, so splits and cross-validation stay clean)."""
    return ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUMERIC),
            ("bin", SimpleImputer(strategy="most_frequent"), BINARY),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL,
            ),
        ],
        verbose_feature_names_out=False,
    )


def ensure_clinical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing clinical columns as NaN (the imputers handle them) and fix dtypes."""
    out = df.copy()
    for c in CLINICAL_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    for c in NUMERIC + BINARY:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def coerce_clinical_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw clinical table: numeric coercion ('N/A' -> NaN), patient_id, label dtype."""
    out = ensure_clinical_columns(df)
    if PATIENT_ID not in out.columns:
        if "id" in out.columns:
            out[PATIENT_ID] = out["id"].astype(str)
        else:
            raise ValueError(f"Need a '{PATIENT_ID}' (or 'id') column so splits can be done per patient.")
    out[PATIENT_ID] = out[PATIENT_ID].astype(str)
    if LABEL in out.columns:
        out[LABEL] = out[LABEL].astype(int)
    return out

"""Clinical-only models with imbalance handling that cannot leak.

Resampling (SMOTE / random oversampling) lives INSIDE an imblearn Pipeline, so it is applied
only to the training portion at fit time and is skipped at predict time. The preprocessor
(imputer/scaler/encoder) and the optional feature selector are fitted on training rows only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

from .data.clinical import build_preprocessor, ensure_clinical_columns

MODELS = ("logreg", "rf", "xgb")
IMBALANCE = ("none", "class_weight", "smote", "random_over")


def _classifier(model: str, imbalance: str, y_train, seed: int):
    cw = imbalance == "class_weight"
    if model == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced" if cw else None, random_state=seed)
    if model == "rf":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=seed,
            class_weight="balanced_subsample" if cw else None,
        )
    if model == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError:  # graceful fallback
            return HistGradientBoostingClassifier(class_weight="balanced" if cw else None, random_state=seed)
        spw = 1.0
        if cw:
            if y_train is None:
                raise ValueError("xgb + class_weight needs y_train to compute scale_pos_weight")
            y_train = np.asarray(y_train)
            spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        return XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=spw, eval_metric="logloss", random_state=seed, n_jobs=1, verbosity=0,
        )
    raise ValueError(f"unknown model {model!r}; choose from {MODELS}")


def make_clinical_pipeline(model: str = "rf", imbalance: str = "class_weight", select_features: bool = False,
                           y_train=None, seed: int = 42) -> ImbPipeline:
    if imbalance not in IMBALANCE:
        raise ValueError(f"unknown imbalance strategy {imbalance!r}; choose from {IMBALANCE}")
    steps = [("pre", build_preprocessor())]
    if select_features:
        steps.append(("select", SelectFromModel(
            RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=seed),
            threshold="median")))
    if imbalance == "smote":
        steps.append(("resample", SMOTE(random_state=seed)))
    elif imbalance == "random_over":
        steps.append(("resample", RandomOverSampler(random_state=seed)))
    steps.append(("clf", _classifier(model, imbalance, y_train, seed)))
    return ImbPipeline(steps)


def feature_matrix(pipe: ImbPipeline, df: pd.DataFrame) -> np.ndarray:
    """Preprocessed (and optionally feature-selected) clinical matrix from an already-fitted pipeline."""
    z = pipe.named_steps["pre"].transform(ensure_clinical_columns(df))
    if "select" in pipe.named_steps:
        z = pipe.named_steps["select"].transform(z)
    return np.asarray(z, dtype=np.float32)


def feature_names(pipe: ImbPipeline) -> list[str]:
    names = list(pipe.named_steps["pre"].get_feature_names_out())
    if "select" in pipe.named_steps:
        names = [n for n, keep in zip(names, pipe.named_steps["select"].get_support()) if keep]
    return names


def predict_clinical(pipe: ImbPipeline, df: pd.DataFrame) -> np.ndarray:
    return pipe.predict_proba(ensure_clinical_columns(df))[:, 1]

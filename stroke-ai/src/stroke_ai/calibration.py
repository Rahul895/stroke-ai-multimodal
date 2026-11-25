"""Probability calibration. Fit on VALIDATION predictions, evaluate on TEST."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .utils import logit


class Calibrator:
    def __init__(self, method: str = "sigmoid"):
        if method not in ("sigmoid", "isotonic"):
            raise ValueError("method must be 'sigmoid' (Platt) or 'isotonic'")
        self.method = method
        self._m = None

    def fit(self, y, p):
        y = np.asarray(y).astype(int)
        if self.method == "sigmoid":
            self._m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logit(p).reshape(-1, 1), y)
        else:
            self._m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(np.asarray(p, float), y)
        return self

    def transform(self, p):
        p = np.asarray(p, float)
        if self.method == "sigmoid":
            return self._m.predict_proba(logit(p).reshape(-1, 1))[:, 1]
        return self._m.predict(p)

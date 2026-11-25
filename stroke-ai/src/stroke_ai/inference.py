"""Scorer: one code path from raw (clinical DataFrame, raw images) to probabilities, used by training,
evaluation, robustness tests, the API and the demo app. StrokePredictor adds calibration,
uncertainty, SHAP and Grad-CAM for single patients."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .calibration import Calibrator
from .clinical_model import feature_matrix, predict_clinical
from .data.clinical import ensure_clinical_columns
from .data.imaging import normalize
from .explain import ClinicalExplainer, grad_cam, overlay_cam, png_bytes
from .fusion import build_net
from .training import predict_net, predict_net_mc

KINDS = ("clinical", "mri", "late_fusion", "feature_fusion", "cross_attention")
DISCLAIMER = ("Research prototype. Not a medical device and not for diagnosis or treatment decisions. "
              "Outputs have not been clinically validated.")


class Scorer:
    def __init__(self, kind: str, clin_pipe=None, net=None, late_w: float | None = None, device: str = "cpu"):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if kind != "mri" and kind != "late_fusion" and kind != "clinical" and net is None:
            raise ValueError("fusion kinds need a net")
        self.kind, self.clin_pipe, self.net = kind, clin_pipe, net
        self.late_w, self.device = late_w, device

    # -- helpers
    def clin_matrix(self, df):
        return feature_matrix(self.clin_pipe, df)

    def _net_inputs(self, df, imgs):
        clin = self.clin_matrix(df) if self.kind in ("feature_fusion", "cross_attention") else None
        return normalize(imgs), clin

    # -- prediction
    def predict(self, df: pd.DataFrame, imgs: np.ndarray | None = None) -> np.ndarray:
        """Raw (uncalibrated) stroke probability per row."""
        if self.kind == "clinical":
            return predict_clinical(self.clin_pipe, df)
        xi, clin = self._net_inputs(df, imgs)
        pn = predict_net(self.net, xi, clin, self.device)
        if self.kind == "late_fusion":
            return self.late_w * predict_clinical(self.clin_pipe, df) + (1 - self.late_w) * pn
        return pn

    def predict_samples(self, df: pd.DataFrame, imgs: np.ndarray | None, T: int = 30) -> np.ndarray:
        """(T, n) probabilities via MC-dropout. Clinical-only models are deterministic -> T=1."""
        if self.kind == "clinical":
            return self.predict(df)[None]
        xi, clin = self._net_inputs(df, imgs)
        s = predict_net_mc(self.net, xi, clin, T, self.device)
        if self.kind == "late_fusion":
            s = self.late_w * predict_clinical(self.clin_pipe, df)[None] + (1 - self.late_w) * s
        return s

    # -- persistence
    def save(self, d) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        meta = {"kind": self.kind, "late_w": self.late_w,
                "net_spec": getattr(self.net, "spec", None)}
        if self.clin_pipe is not None:
            joblib.dump(self.clin_pipe, d / "clin_pipe.joblib")
        if self.net is not None:
            torch.save(self.net.state_dict(), d / "net.pt")
        (d / "scorer.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, d, device: str = "cpu") -> "Scorer":
        d = Path(d)
        meta = json.loads((d / "scorer.json").read_text())
        pipe = joblib.load(d / "clin_pipe.joblib") if (d / "clin_pipe.joblib").exists() else None
        net = None
        if meta["net_spec"] is not None:
            net = build_net(**meta["net_spec"])
            net.load_state_dict(torch.load(d / "net.pt", map_location=device, weights_only=True))
            net.eval()
        return cls(meta["kind"], pipe, net, meta["late_w"], device)


def scorer_gradcam(scorer: Scorer, df_row: pd.DataFrame, img01: np.ndarray) -> np.ndarray:
    """Grad-CAM (H, W) for one patient. For fusion nets it is the image pathway's attention *given* this patient's clinical data."""
    if scorer.kind == "clinical" or scorer.net is None:
        raise ValueError("Grad-CAM needs an image model")
    xi, clin = scorer._net_inputs(df_row, img01[None])
    xi_t = torch.from_numpy(xi)
    clin_t = torch.from_numpy(np.asarray(clin, dtype=np.float32)) if clin is not None else torch.zeros(1, 1)
    scorer.net.eval()
    return grad_cam(lambda x: scorer.net(x, clin_t), scorer.net.backbone.features, xi_t)


class StrokePredictor:
    """Everything needed to score ONE patient: calibrated probability, decision threshold, uncertainty,
    top clinical SHAP factors and a Grad-CAM overlay."""

    def __init__(self, scorer: Scorer, calibrator: Calibrator, decision: dict, explainer: ClinicalExplainer | None):
        self.scorer, self.calibrator, self.decision, self.explainer = scorer, calibrator, decision, explainer
        self.img_size = int(decision["img_size"])

    @classmethod
    def load(cls, model_dir) -> "StrokePredictor":
        d = Path(model_dir)
        scorer = Scorer.load(d)
        calibrator = joblib.load(d / "calibrator.joblib")
        decision = json.loads((d / "decision.json").read_text())
        explainer = None
        if (d / "clinical_explainer.joblib").exists():
            bg = pd.read_csv(d / "shap_background.csv")
            explainer = ClinicalExplainer(joblib.load(d / "clinical_explainer.joblib"), ensure_clinical_columns(bg))
        return cls(scorer, calibrator, decision, explainer)

    def _level(self, std: float) -> str:
        lo, hi = self.decision["uncertainty_edges"]
        return "low" if std <= lo else ("moderate" if std <= hi else "high")

    def predict(self, clinical: dict, img01: np.ndarray, T: int = 30, with_gradcam: bool = True,
                with_shap: bool = True) -> dict:
        df = ensure_clinical_columns(pd.DataFrame([clinical]))
        imgs = np.asarray(img01, dtype=np.float32)[None]
        raw = self.scorer.predict_samples(df, imgs, T)                      # (T, 1)
        cal = self.calibrator.transform(raw.reshape(-1)).reshape(raw.shape)
        prob, std = float(cal.mean()), float(cal.std())
        thr = float(self.decision["threshold"])
        out = {
            "risk_probability": prob,
            "decision_threshold": thr,
            "above_threshold": bool(prob >= thr),
            "assessment": "higher estimated risk" if prob >= thr else "lower estimated risk",
            "uncertainty": {"std": std, "level": self._level(std),
                            "note": "MC-dropout spread; not a calibrated confidence interval. Level is relative to the validation set."},
            "model_kind": self.scorer.kind,
            "disclaimer": DISCLAIMER,
        }
        if with_shap and self.explainer is not None:
            out["top_clinical_factors"] = self.explainer.top_features(df, k=5)
        if with_gradcam and self.scorer.kind != "clinical":
            cam = scorer_gradcam(self.scorer, df, np.asarray(img01, dtype=np.float32))
            out["gradcam_png_base64"] = base64.b64encode(png_bytes(overlay_cam(np.asarray(img01)[0], cam))).decode()
        return out

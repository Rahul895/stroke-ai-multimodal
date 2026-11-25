import base64
import io
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import create_app
from stroke_ai.data.synthetic import make_synthetic_cohort
from stroke_ai.inference import Scorer, StrokePredictor
from stroke_ai.pipeline import RunConfig, run_pipeline


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    cfg = RunConfig(n_samples=260, img_size=32, epochs=3, n_boot=20, mc_samples=3, out_dir=str(out), seed=0)
    run_pipeline(cfg, verbose=False)
    return out


def test_outputs_exist(run_dir):
    for f in ["ablation_test.csv", "calibration.csv", "external_validation.csv", "robustness_test.csv",
              "subgroups_test.csv", "paired_auc_differences.csv", "clinical_grid_val.csv"]:
        assert (run_dir / "results" / f).exists(), f
    for f in ["roc.png", "calibration.png", "shap_clinical.png", "gradcam.png"]:
        assert (run_dir / "figures" / f).exists(), f
    abl = pd.read_csv(run_dir / "results" / "ablation_test.csv")
    assert {"clinical_only", "mri_only", "late_fusion", "feature_fusion", "cross_attention"} <= set(abl.model)
    assert abl.roc_auc.between(0, 1).all()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["proposed"] in {"late_fusion", "feature_fusion", "feature_fusion_featsel", "cross_attention"}


def test_saved_scorer_roundtrip_matches(run_dir):
    df, imgs = make_synthetic_cohort(30, 32, seed=123)
    a = Scorer.load(run_dir / "model")
    b = Scorer.load(run_dir / "model")
    assert np.allclose(a.predict(df, imgs), b.predict(df, imgs))
    assert np.allclose(a.predict(df, imgs), a.predict(df, imgs))          # deterministic at inference


def test_predictor_single_patient(run_dir):
    pred = StrokePredictor.load(run_dir / "model")
    clin = dict(gender="Male", age=70, hypertension=1, heart_disease=0, avg_glucose_level=150.0, bmi=None, smoking_status="smokes")
    res = pred.predict(clin, np.random.rand(1, 32, 32).astype(np.float32), T=5)
    assert 0 <= res["risk_probability"] <= 1 and res["uncertainty"]["level"] in {"low", "moderate", "high"}
    assert len(res["top_clinical_factors"]) == 5 and base64.b64decode(res["gradcam_png_base64"])[:4] == b"\x89PNG"
    assert "not a medical device" in res["disclaimer"].lower()


def _png(arr01):
    buf = io.BytesIO()
    Image.fromarray((arr01[0] * 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def test_api(run_dir):
    client = TestClient(create_app(str(run_dir / "model")))
    assert client.get("/health").json()["model_available"] is True
    _, imgs = make_synthetic_cohort(2, 32, seed=5)
    clin = dict(gender="Female", age=58, hypertension=0, heart_disease=0, avg_glucose_level=98.0, bmi=26.1, smoking_status="never smoked")
    r = client.post("/predict", data={"clinical": json.dumps(clin)}, files={"image": ("s.png", _png(imgs[0]), "image/png")})
    assert r.status_code == 200, r.text
    assert 0 <= r.json()["risk_probability"] <= 1
    bad = dict(clin, age=500)
    assert client.post("/predict", data={"clinical": json.dumps(bad)}, files={"image": ("s.png", _png(imgs[0]), "image/png")}).status_code == 422
    assert client.post("/predict", data={"clinical": json.dumps(clin)}, files={"image": ("s.png", b"not an image", "image/png")}).status_code == 422


def test_api_without_model_returns_503(tmp_path):
    client = TestClient(create_app(str(tmp_path / "nothing")))
    assert client.get("/health").json()["model_available"] is False
    clin = dict(gender="Male", age=50, hypertension=0, heart_disease=0, avg_glucose_level=90.0, smoking_status="Unknown")
    _, imgs = make_synthetic_cohort(2, 32, seed=5)
    assert client.post("/predict", data={"clinical": json.dumps(clin)}, files={"image": ("s.png", _png(imgs[0]), "image/png")}).status_code == 503

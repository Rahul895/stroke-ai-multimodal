import numpy as np
import pandas as pd

from stroke_ai.clinical_model import make_clinical_pipeline
from stroke_ai.data.clinical import ensure_clinical_columns
from stroke_ai.data.splits import patient_level_split
from stroke_ai.data.synthetic import make_synthetic_cohort


def test_synthetic_shapes_and_prevalence():
    df, imgs = make_synthetic_cohort(400, 32, seed=1)
    assert imgs.shape == (400, 1, 32, 32) and imgs.dtype == np.float32
    assert 0.0 <= imgs.min() and imgs.max() <= 1.0
    assert 0.1 < df.stroke.mean() < 0.35
    assert df.patient_id.is_unique


def test_split_is_patient_level_even_with_multiple_scans():
    df, _ = make_synthetic_cohort(300, 16, seed=2)
    dup = pd.concat([df, df.sample(100, random_state=0)], ignore_index=True)   # repeated patients (extra scans)
    sp = patient_level_split(dup, seed=3)
    ids = {k: set(dup.patient_id.iloc[v]) for k, v in sp.items()}
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])
    assert sum(len(v) for v in sp.values()) == len(dup)          # every row assigned exactly once


def test_split_is_stratified():
    df, _ = make_synthetic_cohort(600, 16, seed=4)
    sp = patient_level_split(df, seed=5)
    rates = [df.stroke.iloc[v].mean() for v in sp.values()]
    assert max(rates) - min(rates) < 0.05


def test_preprocessor_fitted_on_training_rows_only():
    df, _ = make_synthetic_cohort(400, 16, seed=6)
    train = df.iloc[:200]
    pipe = make_clinical_pipeline("logreg", "smote", seed=0).fit(ensure_clinical_columns(train), train.stroke)
    scaler = pipe.named_steps["pre"].named_transformers_["num"].named_steps["sc"]
    assert np.isclose(scaler.mean_[0], train.age.mean())              # not the full-data mean
    assert not np.isclose(scaler.mean_[0], df.age.mean())


def test_resampling_is_fit_time_only():
    df, _ = make_synthetic_cohort(300, 16, seed=7)
    pipe = make_clinical_pipeline("rf", "smote", seed=0).fit(ensure_clinical_columns(df.iloc[:200]), df.stroke.iloc[:200])
    test = df.iloc[200:]
    assert len(pipe.predict_proba(ensure_clinical_columns(test))) == len(test)   # SMOTE must not change prediction-time rows


def test_pipeline_handles_missing_values_at_predict_time():
    df, _ = make_synthetic_cohort(300, 16, seed=8)
    pipe = make_clinical_pipeline("xgb", "class_weight", y_train=df.stroke.iloc[:200], seed=0).fit(
        ensure_clinical_columns(df.iloc[:200]), df.stroke.iloc[:200])
    t = df.iloc[200:].copy()
    t.loc[:, ["age", "bmi", "smoking_status"]] = np.nan
    p = pipe.predict_proba(ensure_clinical_columns(t))[:, 1]
    assert np.isfinite(p).all()


def test_normalisation_is_invariant_to_global_brightness_and_gain():
    from stroke_ai.data.imaging import normalize
    _, imgs = make_synthetic_cohort(20, 32, seed=9)
    assert np.allclose(normalize(imgs), normalize(imgs + 0.05), atol=1e-4)
    assert np.allclose(normalize(imgs), normalize(imgs * 0.8), atol=1e-4)
    z = normalize(imgs)
    assert np.allclose(z.mean(axis=(1, 2, 3)), 0, atol=1e-4) and np.allclose(z.std(axis=(1, 2, 3)), 1, atol=1e-3)

def test_nifti_helper_is_label_free():
    from stroke_ai.data.nifti import representative_axial_slice
    v = np.zeros((16, 16, 10), dtype=np.float32)
    v[3:13, 3:13, 4:7] = 2.0
    out = representative_axial_slice(v)
    assert out.shape == (16, 16)
    assert np.isfinite(out).all()

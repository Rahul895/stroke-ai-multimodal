# Stroke-AI — explainable multimodal (clinical + MRI) stroke-risk prototype

A reproducible research prototype: structured clinical data + a brain-MRI slice → calibrated stroke-risk
estimate, with SHAP and Grad-CAM explanations, uncertainty, robustness tests, subgroup analysis, an
ablation study, a REST API and a demo dashboard.

> **Not a medical device.** Nothing here is clinically validated. Do not use it for diagnosis or treatment.

## What is verified, and what is not

| Status | Item |
|---|---|
| ✅ Tested here (28 automated tests, full run reproduces bit-for-bit with the same seed) | data loading, patient-level splits, leakage guards, clinical models, MRI nets, fusion nets, metrics, calibration, bootstrap CIs, robustness, subgroups, SHAP, Grad-CAM, MC-dropout, save/load, FastAPI, Streamlit app starts, CSV loader (PNG + Kaggle-style `N/A`) |
| ⚠️ Run only on **synthetic** data | every number in `docs/example_run/`. The "MRI" images are generated blobs. They prove the code works, **not** that stroke can be predicted. |
| ❌ Not tested | Docker build, GitHub Actions, MLflow logging (`--use-mlflow`), ImageNet-pretrained weights (`pretrained=True` needs internet), GPU |
| ⚠️ Deliberate scope | DVC, DeLong test (bootstrap CIs are used instead), deep ensembles, ViT/Swin, drift/monitoring page, cloud deployment, and full 3-D MRI are outside the current academic prototype scope |


## Teacher-ready demo

See `TEACHER_DEMO.md` for the exact Windows/Linux commands, viva questions, real-NIfTI workflow, and safe claims. `run_demo.bat` and `run_demo.sh` provide a one-command demonstration after the first dependency install.

### Real NIfTI support

The loader now accepts `.nii`/`.nii.gz` directly using a label-free representative axial slice. For repeated experiments, use `scripts/prepare_nifti_dataset.py` to cache `.npy` slices. Install the NIfTI dependency with `pip install -r requirements-real.txt` (or the full `requirements.txt`).

## Quick start

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
pip install -e .
pytest -q                                   # 28 tests, ~15 s
python -m stroke_ai.pipeline --out artifacts   # synthetic demo run, a few minutes on CPU
uvicorn api.main:app --port 8000               # API (docs at /docs)
streamlit run app/streamlit_app.py             # demo + research dashboard
```

## Using REAL data

1. **Read the pairing warning below first.**
2. Build one CSV, one row per (clinical record, scan): `patient_id` (or `id`), `stroke` (0/1), `image_path`
   (relative to the CSV, PNG/JPG/2-D `.npy`/NIfTI `.nii` or `.nii.gz`), and the clinical columns in `src/stroke_ai/data/clinical.py`
   (`gender, age, hypertension, heart_disease, avg_glucose_level, bmi, smoking_status`; `N/A` BMI is fine).
   Change the lists at the top of that file if your columns differ.
3. `python -m stroke_ai.pipeline --data csv --data-csv my.csv --img-size 128 --backbone resnet18`
   Add `--external-csv other_site.csv` for genuine external validation.

### ⚠️ The pairing problem (most important caveat)
Fusion only means something if the clinical record and the MRI come from the **same patient**. Public stroke
datasets are usually either tabular-only (e.g. the popular Kaggle clinical set) **or** images-only. Do **not**
randomly pair a clinical row with an MRI that shares its label: that injects the label into both inputs and
makes fusion look better than it is. Options: use a genuinely paired cohort (hospital data, a paired public
challenge dataset you have verified), or report clinical-only and MRI-only results separately and present
fusion on synthetic pairs purely as a methods demonstration, clearly labelled.

## Pipeline (what `stroke_ai.pipeline` does)

1. Load data → **patient-level** stratified split 70/15/15 (a patient's rows never straddle splits).
2. Clinical grid: {logreg, random forest, XGBoost} × {none, class-weight, SMOTE, random oversampling}.
   Resampling sits inside an `imblearn` pipeline → applied to training rows only. Best by **validation** AUC.
3. MRI network, plus three fusions: late (one weight chosen on validation), feature-level, bidirectional
   cross-modal attention; and a feature-selected variant (selector fitted on train only).
4. Per-scan intensity standardisation (no dataset statistics → no leakage, robust to global brightness/gain shifts).
5. Early stopping, model choice, **decision threshold (Youden)** and Platt calibration all use validation only.
   The proposed model is the best fusion variant **by validation AUC**. The test split is scored once.
6. Ablation table with bootstrap 95 % CIs; paired bootstrap AUC differences; calibration; external validation;
   robustness (missing clinical values, noise/blur/contrast/low-res); subgroup table; MC-dropout uncertainty;
   SHAP and Grad-CAM figures; deployable model in `artifacts/model/`.

Outputs: `artifacts/results/*.csv`, `artifacts/figures/*.png`, `artifacts/summary.json`.

## Example result — SYNTHETIC data, 800 patients, default settings (`docs/example_run/`)

| Model (test, n=120, 24 positives) | ROC-AUC | PR-AUC | Sens. @ 90 % spec. |
|---|---:|---:|---:|
| Clinical only | 0.747 | 0.416 | 0.333 |
| MRI only | 0.812 | 0.580 | 0.750 |
| Late fusion (**proposed**, chosen on validation) | 0.872 (95 % CI 0.775–0.942) | 0.671 | 0.625 |
| Feature fusion | 0.874 | 0.659 | 0.708 |
| Cross-attention | 0.859 | 0.619 | 0.708 |

What this does and does not show: fusion beat clinical-only (paired AUC difference +0.125, CI 0.039–0.209) but its
gain over MRI-only (+0.060, CI −0.013–0.149) is **not** distinguishable from noise at this test size, and the three
fusion variants are statistically indistinguishable. Per-model Brier/ECE in `ablation_test.csv` are for *raw* network
outputs (trained with a class-weighted loss, so not calibrated); use `calibration.csv` for the calibrated model
(Brier 0.115, ECE 0.092). On the shifted "external" synthetic site AUC stayed 0.90 but sensitivity fell to 0.59 and
ECE rose to 0.15. An earlier version using dataset-wide intensity normalisation collapsed completely under a small
brightness shift; the robustness/external checks caught that and it is now guarded by a regression test.

## Layout
```
src/stroke_ai/  data/ (synthetic, clinical, imaging, splits, real) · clinical_model · imaging_model · fusion
                training · metrics · calibration · explain · robustness · fairness · inference · pipeline
api/main.py     FastAPI (/health, /predict)        app/streamlit_app.py  demo + research tabs
tests/          28 tests incl. leakage guards      Dockerfile · .github/workflows/ci.yml
```

## Design notes / known limitations
- Grad-CAM heatmaps are min–max normalised per image; for low-risk cases with no visible finding they show noise.
- For fusion nets Grad-CAM is the image pathway's attention *given that patient's clinical inputs*.
- Uncertainty is MC-dropout spread, not a calibrated interval; the Low/Moderate/High label is relative to the validation set.
- SHAP units differ by model (probability for random forest, log-odds for XGBoost/logistic regression).
- Subgroup rows with < 5 positives are flagged unreliable. The tool reports whether performance *varies*; it does not certify fairness.
- Cross-attention uses no positional embeddings for image tokens (position is only implicit via convolutions).
- Validation data is used for early stopping, model selection, calibration and threshold; with few positives this makes
  validation estimates optimistic. Prefer nested/repeated CV when data is scarce.
- Tested with Python 3.12, torch 2.14, torchvision 0.29, scikit-learn 1.9, pandas 3.1, numpy 2.5, xgboost 3.4,
  shap 0.52, imbalanced-learn 0.14, fastapi 0.165.

# Stroke-AI — Teacher Demo Runbook

## 1. What to say in one sentence

> “This is a research prototype that combines structured clinical information with a brain-MRI representation, compares unimodal and multimodal models, calibrates the selected model, estimates uncertainty, and provides SHAP/Grad-CAM explanations.”

## 2. Safe demo: no medical dataset required

The repository includes a synthetic paired cohort solely to demonstrate that the complete software pipeline works. The generated images are **not MRI** and the numbers must not be presented as medical performance.

### Windows PowerShell

```powershell
cd stroke-ai
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest -q
python -m stroke_ai.pipeline --data synthetic --n-samples 800 --img-size 64 --epochs 10 --n-boot 50 --mc-samples 5 --out-dir artifacts
python scripts/create_demo_data.py
streamlit run app/streamlit_app.py
```

### Linux/macOS

```bash
cd stroke-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest -q
python -m stroke_ai.pipeline --data synthetic --n-samples 800 --img-size 64 --epochs 10 --n-boot 50 --mc-samples 5 --out-dir artifacts
streamlit run app/streamlit_app.py
```

Open the dashboard and use **Demo mode**. Upload one of the generated PNG files from `demo_data/` to demonstrate the inference screen. The screen is a software demonstration only; the generated images are not MRI. Use **Research mode** to show ROC, calibration, SHAP, Grad-CAM, robustness, subgroup and ablation results.

## 3. Real paired MRI data

Install the extra NIfTI dependency:

```bash
pip install -r requirements-real.txt
```

A real CSV must contain:

- `patient_id` (or `id`)
- `stroke` (0/1 according to your explicitly documented research target)
- `image_path` (`.nii`, `.nii.gz`, `.npy`, `.png`, `.jpg`)
- the clinical columns used by the project, or missing columns will be imputed

**Do not randomly pair a clinical row with an MRI from another patient.** The clinical record and MRI must belong to the same patient.

Validate first:

```bash
python scripts/validate_paired_dataset.py data/paired.csv
```

For NIfTI volumes, either run directly:

```bash
python -m stroke_ai.pipeline --data csv --data-csv data/paired.csv --img-size 128 --backbone resnet18 --epochs 25 --out-dir artifacts_real
```

or cache the label-free representative slice:

```bash
python scripts/prepare_nifti_dataset.py data/paired.csv --out-dir data/nifti_cache
python scripts/validate_paired_dataset.py data/nifti_cache/paired.csv
python -m stroke_ai.pipeline --data csv --data-csv data/nifti_cache/paired.csv --img-size 128 --backbone resnet18 --out-dir artifacts_real
```

## 4. What is actually evaluated

- patient-level train/validation/test split
- clinical-only baseline
- MRI-only baseline
- late fusion
- feature fusion
- feature-selected fusion
- cross-modal attention
- validation-selected proposed model
- calibrated probabilities
- bootstrap confidence intervals
- external validation when supplied
- robustness tests
- subgroup analysis
- MC-dropout uncertainty
- SHAP clinical explanation
- Grad-CAM image explanation

## 5. Questions teachers may ask

**Why multimodal?** Clinical variables capture patient context while MRI captures imaging evidence. The project tests whether combining them adds information beyond either modality alone.

**Why patient-level splitting?** If one patient has multiple scans, placing scans in both train and test leaks patient information and inflates performance.

**Why validation-based model selection?** The test set must remain untouched until final evaluation.

**Why calibration?** A model can rank patients well but produce poorly calibrated probabilities. Calibration makes the reported probability more interpretable as an estimate.

**Why SHAP and Grad-CAM?** They provide complementary explanations: SHAP for structured clinical features and Grad-CAM for image regions influencing the network.

**Is this a medical device?** No. It is an academic research prototype and is not clinically validated.

## 6. Claims you should NOT make

Do not say that the model diagnoses stroke, replaces doctors, or is clinically ready. Do not quote synthetic-demo AUC values as medical results. For a real study, report the dataset, target definition, inclusion/exclusion criteria, split strategy, class balance, confidence intervals and external validation results.

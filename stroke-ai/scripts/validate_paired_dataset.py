"""Validate the paired CSV before training.

Usage:
  python scripts/validate_paired_dataset.py data/paired.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stroke_ai.data.clinical import CLINICAL_COLUMNS, LABEL, PATIENT_ID  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--image-root", default=None)
    args = ap.parse_args()
    csv = Path(args.csv)
    df = pd.read_csv(csv)
    required = {PATIENT_ID, LABEL, "image_path"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"FAIL: missing required columns: {missing}")
    if df[PATIENT_ID].isna().any() or df[PATIENT_ID].duplicated().any():
        print("WARNING: patient_id is duplicated. This is allowed for multiple scans, but splits are patient-level.")
    if not set(pd.to_numeric(df[LABEL], errors="coerce").dropna().unique()).issubset({0, 1}):
        raise SystemExit("FAIL: stroke must contain only 0/1 labels.")
    root = Path(args.image_root) if args.image_root else csv.parent
    paths = [root / str(p) for p in df.image_path]
    missing_files = [str(p) for p in paths if not p.exists()]
    print(f"rows: {len(df)}")
    print(f"unique patients: {df[PATIENT_ID].nunique()}")
    print(f"positives: {int((df[LABEL] == 1).sum())}")
    print(f"negatives: {int((df[LABEL] == 0).sum())}")
    print(f"clinical columns present: {', '.join(c for c in CLINICAL_COLUMNS if c in df.columns)}")
    print(f"missing image files: {len(missing_files)}")
    if missing_files:
        print("First missing files:")
        for p in missing_files[:10]:
            print("  ", p)
        raise SystemExit(2)
    if df[PATIENT_ID].duplicated().any():
        print("OK: repeated patient rows detected; the pipeline uses patient-level splitting.")
    print("PASS: paired dataset is structurally valid for the pipeline.")


if __name__ == "__main__":
    main()

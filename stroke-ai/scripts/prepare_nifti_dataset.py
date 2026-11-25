"""Cache representative MRI slices from a metadata CSV.

Input CSV must contain patient_id (or id), stroke, image_path and clinical columns.
image_path points to .nii/.nii.gz files. Output CSV points to cached .npy slices.

Usage:
  python scripts/prepare_nifti_dataset.py raw/metadata.csv --out-dir data/cache
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stroke_ai.data.nifti import nifti_to_npy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--out-dir", default="data/nifti_cache")
    args = ap.parse_args()
    src_csv = Path(args.csv)
    root = Path(args.image_root) if args.image_root else src_csv.parent
    out = Path(args.out_dir)
    df = pd.read_csv(src_csv).copy()
    if "image_path" not in df.columns:
        raise SystemExit("CSV must contain image_path")
    cached = []
    for i, p in enumerate(df.image_path):
        src = root / str(p)
        if not src.exists():
            raise SystemExit(f"Missing MRI: {src}")
        patient = str(df.iloc[i].get("patient_id", df.iloc[i].get("id", i)))
        dst = out / f"{patient}_{i:05d}.npy"
        nifti_to_npy(src, dst)
        cached.append(str(dst.relative_to(out)))
        if (i + 1) % 25 == 0:
            print(f"processed {i + 1}/{len(df)}")
    df["image_path"] = cached
    out.mkdir(parents=True, exist_ok=True)
    result = out / "paired.csv"
    df.to_csv(result, index=False)
    print(f"Wrote {result}")


if __name__ == "__main__":
    main()

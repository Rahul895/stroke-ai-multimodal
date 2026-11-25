"""Create clearly labelled synthetic demo images for the Streamlit screen."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image
from stroke_ai.data.synthetic import make_synthetic_cohort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_data")
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, imgs = make_synthetic_cohort(args.n, 64, seed=123)
    rows = []
    for i, row in df.iterrows():
        p = out / f"{row.patient_id}.png"
        arr = np.clip(imgs[i, 0] * 255, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(p)
        rows.append({**row.to_dict(), "image_path": p.name})
    df.assign(image_path=[f"{x}.png" for x in df.patient_id]).to_csv(out / "clinical_demo.csv", index=False)
    print(f"Created {args.n} synthetic demo images in {out}")
    print("These images are NOT real MRI and must not be presented as clinical evidence.")


if __name__ == "__main__":
    main()

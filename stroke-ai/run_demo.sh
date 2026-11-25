#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
  .venv/bin/python -m pip install -e .
fi
.venv/bin/python -m pytest -q
.venv/bin/python -m stroke_ai.pipeline --data synthetic --n-samples 800 --img-size 64 --epochs 10 --n-boot 50 --mc-samples 5 --out-dir artifacts
.venv/bin/python -m streamlit run app/streamlit_app.py

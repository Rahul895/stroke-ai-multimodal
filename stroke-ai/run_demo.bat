@echo off
setlocal
cd /d %~dp0
if not exist .venv\Scripts\python.exe (
  py -m venv .venv
  .venv\Scripts\python.exe -m pip install --upgrade pip
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  .venv\Scripts\python.exe -m pip install -e .
)
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m stroke_ai.pipeline --data synthetic --n-samples 800 --img-size 64 --epochs 10 --n-boot 50 --mc-samples 5 --out-dir artifacts
.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py

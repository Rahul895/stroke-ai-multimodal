FROM python:3.12-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
# CPU-only PyTorch keeps the image small.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
COPY api ./api
COPY app ./app
RUN pip install --no-deps -e .
# Mount or bake a trained model at /app/artifacts/model (run `python -m stroke_ai.pipeline` first).
ENV STROKE_MODEL_DIR=/app/artifacts/model
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

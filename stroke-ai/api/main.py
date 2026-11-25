"""FastAPI service.  Run:  STROKE_MODEL_DIR=artifacts/model uvicorn api.main:app --port 8000

POST /predict   multipart form: `clinical` = JSON string, `image` = PNG/JPG slice.
"""
from __future__ import annotations

import io
import os
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, ValidationError

from stroke_ai.data.imaging import load_image
from stroke_ai.inference import DISCLAIMER, StrokePredictor

MAX_IMAGE_BYTES = 10 * 1024 * 1024


class ClinicalInput(BaseModel):
    gender: str = Field(..., pattern="^(Male|Female)$")
    age: float = Field(..., ge=0, le=120)
    hypertension: int = Field(..., ge=0, le=1)
    heart_disease: int = Field(..., ge=0, le=1)
    avg_glucose_level: float = Field(..., ge=30, le=500)
    bmi: Optional[float] = Field(None, ge=10, le=80)
    smoking_status: str = Field("Unknown", pattern="^(never smoked|formerly smoked|smokes|Unknown)$")


def create_app(model_dir: str | None = None) -> FastAPI:
    app = FastAPI(title="Stroke-AI decision-support prototype", version="0.1.0", description=DISCLAIMER)
    app.state.model_dir = model_dir or os.environ.get("STROKE_MODEL_DIR", "artifacts/model")
    app.state.predictor = None

    def predictor() -> StrokePredictor:
        if app.state.predictor is None:
            try:
                app.state.predictor = StrokePredictor.load(app.state.model_dir)
            except FileNotFoundError:
                raise HTTPException(503, f"No trained model in '{app.state.model_dir}'. Run: python -m stroke_ai.pipeline")
        return app.state.predictor

    @app.get("/health")
    def health():
        from pathlib import Path
        return {"status": "ok", "model_available": (Path(app.state.model_dir) / "scorer.json").exists()}

    @app.post("/predict")
    async def predict(clinical: str = Form(...), image: UploadFile = File(...)):
        try:
            data = ClinicalInput.model_validate_json(clinical)
        except ValidationError as e:
            raise HTTPException(422, detail=e.errors(include_url=False, include_context=False))
        raw = await image.read()
        if len(raw) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "Image too large (max 10 MB).")
        pred = predictor()
        try:
            img = load_image(io.BytesIO(raw), pred.img_size)
        except Exception:
            raise HTTPException(422, "Could not read the image. Upload a PNG or JPG slice.")
        return pred.predict(data.model_dump(), img)

    return app


app = create_app()
